import json
from pathlib import Path

from ofc_regular.hu_turn1_topk_confirm import parse_hu_turn1_topk_confirm_configs


def test_stage18_p1_acceptance_config_is_locked_and_selective():
    config = json.loads(
        Path("configs/hu_turn1_stage18_p1_selective_override.json").read_text(
            encoding="utf-8"
        )
    )

    assert config["status"] == "p1_accepted_for_downstream_experiments_not_current_default"
    assert config["ai_profile"] == "stage18_p1"
    assert config["t1_p1_fixed"] is True
    assert config["production_default"] is False
    assert config["current_profile_changed"] is False
    assert config["selective_override_only"] is True
    assert config["full_replacement_enabled"] is False
    assert config["allowed_seats"] == ["first"]
    assert config["models"] == {
        "candidate": "models/hu_turn1_stage18_first1801_mc32_hgb_abs_aug3.pkl",
        "safe_selector": "models/hu_turn1_stage18_highmc_safe_selector_a_meta_only.pkl",
    }

    parsed = parse_hu_turn1_topk_confirm_configs(config["runtime"]["config_string"])[0]
    assert parsed.top_k == 5
    assert parsed.mc_samples == 8
    assert parsed.min_delta == 3.0
    assert parsed.confirm_mc_samples == 32
    assert parsed.confirm_se_multiplier == 1.5
    assert parsed.min_predicted_delta == 1.5
    assert parsed.safe_selector_threshold == 0.7
    assert parsed.allowed_seats == ("first",)

    assert config["continuations"]["t2"]["profile"] == "stage9f_p2"
    assert config["continuations"]["t3"]["profile"] == "stage7_m5_r10"
    assert config["continuations"]["t3"]["hu_turn3_min_margin"] == 5.0
    assert config["continuations"]["t3"]["hu_turn3_reference_min_margin"] == 10.0
    assert config["safety"]["confirm_delta_is_gate_diagnostic_only"] is True
    assert config["safety"]["performance_metric"] == (
        "realized_seat_swap_counterfactual_delta"
    )

    evidence = config["acceptance_evidence"]
    assert evidence["paired_seeds"] == 250000
    assert evidence["decisions"] == 500000
    assert evidence["realized_fires"] >= 250
    assert evidence["realized_per_fire_ci95_low"] > 0.0
    assert evidence["estimated_ev_per_decision_ci95_low"] > 0.0
    assert evidence["whole_hand_ev"] > 0.0
    assert evidence["invalid_overrides"] == 0
    assert evidence["non_fired_counterfactual_nonzero_count"] == 0
    assert evidence["non_fired_final_mismatch_count"] == 0
    assert evidence["non_fired_final_unknown_count"] == 0
    assert evidence["p95_loss"] <= 25.0
    assert evidence["p99_loss"] <= 40.0
    assert evidence["max_loss"] <= 50.0
    assert evidence["acceptance_passed"] is True
    smoke = config["post_wiring_smoke"]
    assert smoke["t1_decisions"] == 20
    assert smoke["runtime_profile_rows"] == 20
    assert smoke["runtime_status_rows"] == 20
    assert smoke["second_seat_not_allowed_rows"] == 10
    assert smoke["replay_ready_rows"] == 20
    assert smoke["non_fired_counterfactual_nonzero_count"] == 0
    assert smoke["non_fired_final_mismatch_count"] == 0
    assert len(config["model_integrity"]["candidate_sha256"]) == 64
    assert len(config["model_integrity"]["safe_selector_sha256"]) == 64
    assert config["rollback"] == {
        "profile": "stage9f_p2",
        "preset": "stage18_p1_off",
    }


def test_stage18_p1_presets_keep_off_rollback_and_exclude_unvalidated_modes():
    config = json.loads(
        Path("configs/hu_turn1_stage18_p1_presets.json").read_text(encoding="utf-8")
    )
    presets = {item["name"]: item for item in config["presets"]}

    assert config["current_profile_changed"] is False
    assert presets["stage18_p1_off"] == {
        "name": "stage18_p1_off",
        "ai_profile": "stage9f_p2",
        "enabled": False,
        "production_default": False,
        "fallback_policy": "stage9f_p2",
    }
    p1 = presets["stage18_p1"]
    assert p1["ai_profile"] == "stage18_p1"
    assert p1["enabled"] is True
    assert p1["validation_default"] is True
    assert p1["production_default"] is False
    assert p1["allowed_seats"] == ["first"]
    assert p1["selective_override_only"] is True
    assert p1["full_replacement_enabled"] is False
    assert p1["fallback_policy"] == "stage9f_p2"

    excluded = {item["name"] for item in config["production_excluded"]}
    assert excluded == {"second_seat", "full_replacement", "current_profile"}
