import json
import shutil
from pathlib import Path

from ofc_regular.evaluate_hu_turn2_stage8b_topk_mc_rerank import parse_topk_configs
from ofc_regular.verify_hu_turn2_stage9f_guarded_preset import verify_guarded_preset


def test_stage9f_preflight_config_is_first_seat_selective_override_only():
    config = json.loads(
        Path("configs/hu_turn2_stage9f_cse1p5_firstseat_preflight.json").read_text(
            encoding="utf-8"
        )
    )

    assert config["production_default"] is False
    assert config["production_p2_fixed"] is False
    assert config["full_replacement_enabled"] is False
    assert config["seat_scope"]["allowed_seats"] == ["first"]
    assert config["seat_scope"]["second_seat_enabled"] is False
    assert config["t3_continuation"]["profile"] == "stage7_m5_r10"
    assert config["t3_continuation"]["actual_runtime_threshold_wiring_required"] is True
    assert config["runtime"]["config_string"] == (
        "k3/mc16/d0/se0/confirm32/cse1.5/pd0/seat=first/bygate_delta"
    )
    assert config["runtime"]["confirm_se_max"] is None
    assert config["safety"]["missing_model_fallback_requires_explicit_flag"] is True
    assert config["safety"]["missing_model_fallback_cli_flag"] == (
        "--allow-missing-stage8b-model-fallback"
    )
    assert config["primary_metric"]["source"] == "realized_fired_whole_game_delta"
    assert config["primary_metric"]["confirm_delta_role"] == "gate_diagnostic_only"
    assert config["primary_metric"]["confirm_delta_performance_claim_allowed"] is False
    assert config["safety"]["actual_t3_runtime_must_match_t3_continuation"] is True
    assert config["recommended_metrics"]["profile_canary100_realized_m5r10_non_fired_nonzero_count"] == 0
    assert config["recommended_metrics"]["profile_canary1250_realized_m5r10_non_fired_nonzero_count"] == 0
    assert config["recommended_metrics"]["profile_canary1250_realized_m5r10_override_count"] == 38
    assert config["recommended_metrics"]["profile_canary3500_realized_m5r10_non_fired_nonzero_count"] == 0
    assert config["recommended_metrics"]["profile_canary3500_realized_m5r10_override_count"] == 109
    assert config["recommended_metrics"]["profile_canary3500_realized_m5r10_per_fire_ci95_low"] < 0.0
    assert config["recommended_metrics"]["profile_canary3500_realized_m5r10_max_loss"] > 30.0
    assert config["recommended_metrics"]["profile_canary3500_guard_confirm_se2p5_fired_count"] == 81
    assert config["recommended_metrics"]["profile_canary3500_guard_confirm_se2p5_per_fire_ci95_low"] < 0.0
    assert config["recommended_metrics"]["profile_canary3500_guard_confirm_se2p5_p95_loss"] < (
        config["recommended_metrics"]["profile_canary3500_realized_m5r10_p95_loss"]
    )
    assert config["recommended_metrics"]["csemax2p5_profile_canary3500_non_fired_nonzero_count"] == 0
    assert config["recommended_metrics"]["csemax2p5_profile_canary3500_override_count"] == 64
    assert config["recommended_metrics"]["csemax2p5_profile_canary3500_p95_loss"] < (
        config["recommended_metrics"]["profile_canary3500_realized_m5r10_p95_loss"]
    )
    assert config["recommended_metrics"]["csemax2p5_profile_canary3500_guard_confirm_z2_per_fire_ci95_low"] > 0.0
    assert config["recommended_metrics"]["cse2_csemax2p5_profile_canary3500_non_fired_nonzero_count"] == 0
    assert config["recommended_metrics"]["cse2_csemax2p5_profile_canary3500_override_count"] == 46
    assert config["recommended_metrics"]["cse2_csemax2p5_profile_canary3500_p95_loss"] <= (
        config["recommended_metrics"]["csemax2p5_profile_canary3500_p95_loss"]
    )
    assert config["recommended_metrics"]["cse2_csemax2p5_profile_canary3500_guard_confirm_se2_per_fire_ci95_low"] > 0.0
    assert config["recommended_metrics"]["cse2_csemax2p5_profile_canary3500_guard_confirm_se2_max_loss"] <= 2.0
    assert config["recommended_metrics"]["cse2_csemax2_profile_canary3500_non_fired_nonzero_count"] == 0
    assert config["recommended_metrics"]["cse2_csemax2_profile_canary3500_override_count"] == 39
    assert config["recommended_metrics"]["cse2_csemax2_profile_canary3500_ev_per_hand"] > 0.0
    assert config["recommended_metrics"]["cse2_csemax2_profile_canary3500_ci95_low"] > 0.0
    assert config["recommended_metrics"]["cse2_csemax2_profile_canary3500_per_fire_ci95_low"] > 0.0
    assert config["recommended_metrics"]["cse2_csemax2_profile_canary3500_p95_loss"] <= 2.0
    assert config["recommended_metrics"]["cse2_csemax2_profile_canary10000_non_fired_nonzero_count"] == 0
    assert config["recommended_metrics"]["cse2_csemax2_profile_canary10000_override_count"] == 90
    assert config["recommended_metrics"]["cse2_csemax2_profile_canary10000_ev_per_hand"] > 0.0
    assert config["recommended_metrics"]["cse2_csemax2_profile_canary10000_ci95_low"] > 0.0
    assert config["recommended_metrics"]["cse2_csemax2_profile_canary10000_per_fire_ci95_low"] > 0.0
    assert config["recommended_metrics"]["cse2_csemax2_profile_canary10000_max_loss"] > 30.0
    assert (
        config["recommended_metrics"]["cse2_csemax2_profile_canary10000_p95_loss"]
        > config["recommended_metrics"]["cse2_csemax2_profile_canary3500_p95_loss"]
    )
    assert config["recommended_metrics"]["cse2_csemax2_profile_canary10000_guard_confirm_z2p5_per_fire_ci95_low"] > 0.0
    assert config["recommended_metrics"]["cse2_csemax2_tail_guard_targets_10000_total"] == 114
    assert config["recommended_metrics"]["cse2_csemax2_tail_guard_targets_10000_replay_ready"] == 114
    assert config["recommended_metrics"]["cse2_csemax2_tail_guard_targets_10000_source_fired_realized"] == 90
    assert config["recommended_metrics"]["cse2_csemax2_tail_guard_targets_10000_tail_losses"] == 7
    assert config["recommended_metrics"]["cse2_csemax2_tail_guard_targets_10000_severe_tail_losses"] == 5
    assert config["recommended_metrics"]["cse2_csemax2_tail_guard_targets_10000_positive_controls"] == 28
    assert config["recommended_metrics"]["cse2_csemax2_tail_guard_targets_10000_zero_controls"] == 52
    assert config["recommended_metrics"]["cse2_csemax2_tail_guard_targets_10000_confirm_z_boundary"] == 27
    assert config["recommended_metrics"]["cse2_csemax2_tail_guard_replay_readiness_10000_ready"] == 114
    assert config["recommended_metrics"]["cse2_csemax2_tail_guard_replay_readiness_10000_total"] == 114
    assert config["recommended_metrics"]["cse2_csemax2_tail_guard_replay_tail_loss_mc512_targets"] == 7
    assert config["recommended_metrics"]["cse2_csemax2_tail_guard_replay_tail_loss_mc512_successes"] == 7
    assert config["recommended_metrics"]["cse2_csemax2_tail_guard_replay_tail_loss_mc512_failures"] == 0
    assert config["recommended_metrics"]["cse2_csemax2_tail_guard_replay_tail_loss_mc512_gain_mean"] > 0.0
    assert config["recommended_metrics"]["cse2_csemax2_tail_guard_replay_tail_loss_mc512_gain_min"] < 0.0
    assert config["recommended_metrics"]["cse2_csemax2_tail_guard_replay_tail_loss_mc512_high_mc_losses"] == 1
    assert config["recommended_metrics"]["cse2_csemax2_tail_guard_replay_tail_loss_mc512_lcb95_positive"] == 5
    assert config["recommended_metrics"]["cse2_csemax2_tail_guard_replay_tail_loss_mc512_sign_flips"] == 6
    assert config["recommended_metrics"]["cse2_csemax2_tail_guard_labels_tail_loss_mc512_rows"] == 7
    assert config["recommended_metrics"]["cse2_csemax2_tail_guard_labels_tail_loss_mc512_hard_negative"] == 1
    assert config["recommended_metrics"]["cse2_csemax2_tail_guard_labels_tail_loss_mc512_safe_positive"] == 5
    assert config["recommended_metrics"]["cse2_csemax2_tail_guard_labels_tail_loss_mc512_gray"] == 1
    assert config["recommended_metrics"]["cse2_csemax2_tail_guard_replay_all_mc512_gcp_targets"] == 114
    assert config["recommended_metrics"]["cse2_csemax2_tail_guard_replay_all_mc512_gcp_successes"] == 114
    assert config["recommended_metrics"]["cse2_csemax2_tail_guard_replay_all_mc512_gcp_failures"] == 0
    assert config["recommended_metrics"]["cse2_csemax2_tail_guard_replay_all_mc512_gcp_raw_labels"] == 114
    assert config["recommended_metrics"]["cse2_csemax2_tail_guard_replay_all_mc512_gcp_raw_hard_negative"] == 8
    assert config["recommended_metrics"]["cse2_csemax2_tail_guard_replay_all_mc512_gcp_raw_safe_positive"] == 72
    assert config["recommended_metrics"]["cse2_csemax2_tail_guard_replay_all_mc512_gcp_raw_gray"] == 34
    assert config["recommended_metrics"]["cse2_csemax2_tail_guard_replay_all_mc512_gcp_dedup_labels"] == 89
    assert config["recommended_metrics"]["cse2_csemax2_tail_guard_replay_all_mc512_gcp_duplicate_source_rows"] == 25
    assert config["recommended_metrics"]["cse2_csemax2_tail_guard_replay_all_mc512_gcp_dedup_hard_negative"] == 5
    assert config["recommended_metrics"]["cse2_csemax2_tail_guard_replay_all_mc512_gcp_dedup_safe_positive"] == 57
    assert config["recommended_metrics"]["cse2_csemax2_tail_guard_replay_all_mc512_gcp_dedup_gray"] == 27
    assert config["recommended_metrics"]["cse2_csemax2_profile_canary3500_guard_rank1_per_fire_ci95_low"] > 0.0
    assert config["recommended_metrics"]["cse2_csemax2_profile_canary3500_guard_rank1_max_loss"] <= 2.0
    assert config["recommended_metrics"]["cse2_csemax2_rank1_partial_non_fired_nonzero_count"] == 0
    assert config["recommended_metrics"]["cse2_csemax2_rank1_partial_override_count"] == 28
    assert config["recommended_metrics"]["cse2_csemax2_rank1_partial_estimated_ev_per_decision"] < 0.0
    assert config["recommended_metrics"]["cse2_csemax2_rank1_partial_per_fire_ci95_low"] < 0.0
    assert config["recommended_metrics"]["cse2_csemax2_rank1_partial_max_loss"] > 20.0
    assert "csemax3_validation_ev_per_hand" not in config["recommended_metrics"]
    assert (
        config["recommended_metrics"]["stage9f_c4_nonoverlap_cse1p5_ev_per_hand"]
        > config["recommended_metrics"]["stage9f_c4_nonoverlap_cse1p5_csemax3_ev_per_hand"]
    )
    assert (
        config["evaluation_artifacts"]["model_load_failure_fallback_smoke"]
        == "outputs/evals/hu_turn2_stage9f_preflight_missing_model_fallback_smoke/"
    )
    assert config["evaluation_artifacts"]["profile_canary100_superseded_t3_threshold_mismatch"] == (
        "outputs/evals/hu_turn2_stage9f_profile_canary100/"
    )
    assert config["evaluation_artifacts"]["profile_canary100_realized_m5r10"] == (
        "outputs/evals/hu_turn2_stage9f_profile_canary100_realized_m5r10/"
    )
    assert config["evaluation_artifacts"]["profile_canary1250_realized_m5r10"] == (
        "outputs/evals/hu_turn2_stage9f_profile_canary1250_realized_m5r10/"
    )
    assert config["evaluation_artifacts"]["profile_canary3500_realized_m5r10"] == (
        "outputs/evals/hu_turn2_stage9f_profile_canary3500_realized_m5r10/"
    )
    assert config["evaluation_artifacts"]["profile_canary3500_tail_loss_audit"] == (
        "outputs/evals/hu_turn2_stage9f_profile_canary3500_realized_m5r10/tail_loss_audit/"
    )
    assert config["evaluation_artifacts"]["csemax2p5_profile_smoke"] == (
        "outputs/evals/hu_turn2_stage9f_csemax2p5_profile_smoke1/"
    )
    assert config["evaluation_artifacts"]["csemax2p5_profile_canary3500"] == (
        "outputs/evals/hu_turn2_stage9f_csemax2p5_profile_canary3500_realized_m5r10/"
    )
    assert config["evaluation_artifacts"]["cse2_csemax2p5_profile_smoke"] == (
        "outputs/evals/hu_turn2_stage9f_cse2_csemax2p5_profile_smoke1/"
    )
    assert config["evaluation_artifacts"]["cse2_csemax2p5_profile_canary3500"] == (
        "outputs/evals/hu_turn2_stage9f_cse2_csemax2p5_profile_canary3500_realized_m5r10/"
    )
    assert config["evaluation_artifacts"]["cse2_csemax2_profile_smoke"] == (
        "outputs/evals/hu_turn2_stage9f_cse2_csemax2_profile_smoke1/"
    )
    assert config["evaluation_artifacts"]["cse2_csemax2_profile_canary3500"] == (
        "outputs/evals/hu_turn2_stage9f_cse2_csemax2_profile_canary3500_realized_m5r10/"
    )
    assert config["evaluation_artifacts"]["cse2_csemax2_profile_canary10000"] == (
        "outputs/evals/hu_turn2_stage9f_cse2_csemax2_profile_canary10000_realized_m5r10/"
    )
    assert config["evaluation_artifacts"]["cse2_csemax2_tail_guard_targets_10000"] == (
        "outputs/training/hu_turn2_stage9f_cse2_csemax2_tail_guard_targets_10000/"
    )
    assert config["evaluation_artifacts"]["cse2_csemax2_tail_guard_replay_readiness_10000"] == (
        "outputs/evals/hu_turn2_stage9f_tail_guard_replay_readiness_10000/"
    )
    assert config["evaluation_artifacts"]["cse2_csemax2_tail_guard_replay_tail_loss_mc512"] == (
        "outputs/evals/hu_turn2_stage9f_tail_guard_replay_tail_loss_mc512/"
    )
    assert config["evaluation_artifacts"]["cse2_csemax2_tail_guard_labels_tail_loss_mc512"] == (
        "outputs/training/hu_turn2_stage9f_tail_guard_labels_tail_loss_mc512/"
    )
    assert config["evaluation_artifacts"]["cse2_csemax2_tail_guard_replay_all_mc512_gcp"] == (
        "outputs/evals/hu_turn2_stage9f_tail_guard_replay_all_mc512_gcp/"
    )
    assert config["evaluation_artifacts"]["cse2_csemax2_tail_guard_replay_partial_smoke_mc16"] == (
        "outputs/evals/hu_turn2_stage9f_tail_guard_replay_partial_smoke_mc16_limit1/"
    )
    assert config["evaluation_artifacts"]["cse2_csemax2_rank1_profile_smoke"] == (
        "outputs/evals/hu_turn2_stage9f_cse2_csemax2_rank1_profile_smoke1/"
    )
    assert config["evaluation_artifacts"]["cse2_csemax2_rank1_profile_partial_validation"] == (
        "outputs/evals/hu_turn2_stage9f_cse2_csemax2_rank1_profile_canary10000_realized_m5r10/"
    )

    parsed = parse_topk_configs(config["runtime"]["config_string"])[0]
    assert parsed.top_k == 3
    assert parsed.mc_samples == 16
    assert parsed.confirm_mc_samples == 32
    assert parsed.confirm_se_multiplier == 1.5
    assert parsed.max_confirm_se is None
    assert parsed.allowed_seats == ("first",)


def test_stage9f_canary_presets_have_no_production_default_and_reject_csemax3():
    config = json.loads(
        Path("configs/hu_turn2_stage9f_canary_presets.json").read_text(encoding="utf-8")
    )
    production_defaults = [
        preset for preset in config["presets"] if preset.get("production_default") is True
    ]
    validation_defaults = [
        preset for preset in config["presets"] if preset.get("validation_default") is True
    ]

    assert production_defaults == []
    assert [preset["name"] for preset in validation_defaults] == [
        "stage9f_cse2_csemax2_bothseat"
    ]
    assert validation_defaults[0]["ai_profile"] == "stage9f_cse2_csemax2_bothseat"
    assert any(
        item["name"] == "stage9f_cse1p5_csemax3_firstseat"
        for item in config["production_excluded"]
    )
    assert any(
        item["name"] == "stage9f_cse2_csemax2_rank1_firstseat"
        for item in config["production_excluded"]
    )
    assert not any(
        item["name"] == "all_second_seat_presets" for item in config["production_excluded"]
    )
    assert any(item["name"] == "production_runtime" for item in config["production_excluded"])
    assert any(item["name"] == "full_replacement" for item in config["production_excluded"])

    off = next(preset for preset in config["presets"] if preset["name"] == "stage9f_off")
    assert off["runtime"]["enabled"] is False
    assert off["runtime"]["fallback_policy"] == "baseline_turn2"

    csemax = next(
        preset
        for preset in config["presets"]
        if preset["name"] == "stage9f_cse1p5_csemax3_firstseat"
    )
    assert csemax["status"] == "rejected_guard"
    assert csemax["ai_profile"] is None
    parsed = parse_topk_configs(csemax["runtime"]["config_string"])[0]
    assert parsed.max_confirm_se == 3.0

    csemax2p5 = next(
        preset
        for preset in config["presets"]
        if preset["name"] == "stage9f_cse1p5_csemax2p5_firstseat"
    )
    assert csemax2p5["status"] == "tail_guard_candidate_from_profile_canary3500"
    assert csemax2p5["ai_profile"] == "stage9f_cse1p5_csemax2p5_firstseat"
    assert csemax2p5["production_default"] is False
    parsed = parse_topk_configs(csemax2p5["runtime"]["config_string"])[0]
    assert parsed.max_confirm_se == 2.5

    cse2max2p5 = next(
        preset
        for preset in config["presets"]
        if preset["name"] == "stage9f_cse2_csemax2p5_firstseat"
    )
    assert cse2max2p5["status"] == "tail_guard_candidate_from_csemax2p5_retrospective_sweep"
    assert cse2max2p5["ai_profile"] == "stage9f_cse2_csemax2p5_firstseat"
    assert cse2max2p5["production_default"] is False
    parsed = parse_topk_configs(cse2max2p5["runtime"]["config_string"])[0]
    assert parsed.confirm_se_multiplier == 2.0
    assert parsed.max_confirm_se == 2.5

    cse2max2 = next(
        preset
        for preset in config["presets"]
        if preset["name"] == "stage9f_cse2_csemax2_firstseat"
    )
    assert cse2max2["status"] == "independent_profile_canary10000_positive_tail_risk_validation_only"
    assert cse2max2["ai_profile"] == "stage9f_cse2_csemax2_firstseat"
    assert cse2max2["production_default"] is False
    parsed = parse_topk_configs(cse2max2["runtime"]["config_string"])[0]
    assert parsed.confirm_se_multiplier == 2.0
    assert parsed.max_confirm_se == 2.0

    bothseat = next(
        preset
        for preset in config["presets"]
        if preset["name"] == "stage9f_cse2_csemax2_bothseat"
    )
    assert bothseat["status"] == "bothseat_production_like_canary30000_positive_promotion_review_ready"
    assert bothseat["ai_profile"] == "stage9f_cse2_csemax2_bothseat"
    assert bothseat["production_default"] is False
    assert bothseat["validation_default"] is True
    assert bothseat["experiment_only"] is False
    assert bothseat["runtime"]["allowed_seats"] == ["first", "second"]
    assert bothseat["runtime"]["full_replacement_enabled"] is False
    parsed = parse_topk_configs(bothseat["runtime"]["config_string"])[0]
    assert parsed.confirm_se_multiplier == 2.0
    assert parsed.max_confirm_se == 2.0
    assert parsed.first_min_confirm_delta == 0.0
    assert parsed.second_min_confirm_delta == 0.0
    assert parsed.min_predicted_delta == 0.0
    assert parsed.allowed_seats == ("first", "second")

    cse2max2rank1 = next(
        preset
        for preset in config["presets"]
        if preset["name"] == "stage9f_cse2_csemax2_rank1_firstseat"
    )
    assert cse2max2rank1["status"] == "rejected_after_partial_independent_validation_tail_loss"
    assert cse2max2rank1["ai_profile"] == "stage9f_cse2_csemax2_rank1_firstseat"
    assert cse2max2rank1["production_default"] is False
    parsed = parse_topk_configs(cse2max2rank1["runtime"]["config_string"])[0]
    assert parsed.confirm_se_multiplier == 2.0
    assert parsed.max_confirm_se == 2.0
    assert parsed.candidate_ev_rank_max == 1


def test_stage9f_bothseat_limited_canary_rollout_is_rollback_safe():
    config = json.loads(
        Path(
            "configs/hu_turn2_stage9f_cse2_csemax2_bothseat_limited_canary_rollout.json"
        ).read_text(encoding="utf-8")
    )

    assert config["status"] == "production_like_canary_pass_promotion_review_ready"
    assert config["production_default"] is False
    assert config["production_p2_fixed"] is False
    assert config["validation_default"] is True
    assert config["canary_enabled"] is True
    assert config["full_replacement_enabled"] is False
    assert config["ai_profile"] == "stage9f_cse2_csemax2_bothseat"
    assert config["rollback"]["preset"] == "stage9f_off"
    assert config["rollback"]["config_file"] == "configs/hu_turn2_stage9f_canary_presets.json"

    assert config["t3_continuation"]["profile"] == "stage7_m5_r10"
    assert config["t3_continuation"]["hu_turn3_min_margin"] == 5.0
    assert config["t3_continuation"]["hu_turn3_reference_min_margin"] == 10.0
    assert config["t3_continuation"]["full_replacement_enabled"] is False

    assert config["runtime"]["enabled"] is True
    assert config["runtime"]["allowed_seats"] == ["first", "second"]
    assert config["runtime"]["full_replacement_enabled"] is False
    parsed = parse_topk_configs(config["runtime"]["config_string"])[0]
    assert parsed.top_k == 3
    assert parsed.mc_samples == 16
    assert parsed.confirm_mc_samples == 32
    assert parsed.confirm_se_multiplier == 2.0
    assert parsed.max_confirm_se == 2.0
    assert parsed.first_min_confirm_delta == 0.0
    assert parsed.second_min_confirm_delta == 0.0
    assert parsed.min_predicted_delta == 0.0
    assert parsed.allowed_seats == ("first", "second")

    assert config["safety"]["baseline_turn2_is_default"] is True
    assert config["safety"]["stage9f_selective_override_only"] is True
    assert config["safety"]["runtime_logs_must_be_replay_ready"] is True
    assert config["primary_metric"]["source"] == "realized_fired_whole_game_delta"
    assert config["primary_metric"]["confirm_delta_performance_claim_allowed"] is False

    evidence = config["current_evidence"]
    monitoring = config["monitoring"]
    assert evidence["c4_realized_fires"] == 4000
    assert evidence["c4_per_fire_ci95_low"] > 0.0
    assert evidence["c4_non_fired_nonzero_count"] == 0
    assert evidence["profile_canary2500_realized_fires"] == 52
    assert evidence["profile_canary2500_per_fire_ci95_low"] > 0.0
    assert evidence["profile_canary2500_first_per_fire_ci95_low"] > 0.0
    assert evidence["profile_canary2500_second_per_fire_ci95_low"] > 0.0
    assert evidence["profile_canary2500_non_fired_nonzero_count"] == 0
    assert evidence["production_like_canary30000_decisions"] == 60000
    assert evidence["production_like_canary30000_realized_fires"] >= 500
    assert evidence["production_like_canary30000_per_fire_ci95_low"] > 0.0
    assert evidence["production_like_canary30000_first_per_fire_ci95_low"] > 0.0
    assert evidence["production_like_canary30000_second_per_fire_ci95_low"] > 0.0
    assert evidence["production_like_canary30000_non_fired_nonzero_count"] == 0
    assert evidence["production_like_canary30000_replay_ready"] == 60000
    assert evidence["production_like_canary30000_p95_loss"] <= monitoring["warning_thresholds"]["p95_fired_loss"]
    assert evidence["production_like_canary30000_max_loss"] < monitoring["rollback_thresholds"]["max_fired_loss"]
    assert evidence["production_like_canary30000_latency_p95_ms"] > monitoring["warning_thresholds"]["latency_p95_ms"]
    assert evidence["production_like_canary30000_latency_p95_ms"] < monitoring["rollback_thresholds"]["latency_p95_ms"]
    assert evidence["production_like_canary30000_latency_warning"] is True

    assert monitoring["limited_canary_minimum_paired_decisions"] == 5000
    assert monitoring["limited_canary_minimum_fired_decisions"] == 50
    assert monitoring["promotion_minimum_fired_decisions"] >= 500
    assert monitoring["required"]["missing_replay_fields"] == 0
    assert monitoring["required"]["non_fired_nonzero_count"] == 0
    assert monitoring["required"]["illegal_candidate_overrides"] == 0
    assert monitoring["rollback_thresholds"]["per_fire_ci95_low"] == 0.0
    assert monitoring["rollback_thresholds"]["first_per_fire_ci95_low"] == 0.0
    assert monitoring["rollback_thresholds"]["second_per_fire_ci95_low"] == 0.0
    assert monitoring["rollback_thresholds"]["p95_fired_loss"] >= (
        monitoring["warning_thresholds"]["p95_fired_loss"]
    )
    assert monitoring["rollback_thresholds"]["max_fired_loss"] > evidence["profile_canary2500_max_loss"]

    assert config["promotion_requirements"]["production_default"].startswith("Review-Go")
    assert config["promotion_requirements"]["p2_fixed"].startswith("Review-Go")
    assert config["promotion_requirements"]["teacher_50k"] == "No-Go"
    assert config["promotion_requirements"]["turn1_training"] == "No-Go"


def test_stage9f_bothseat_guarded_production_preset_is_explicit_opt_in_only():
    config = json.loads(
        Path(
            "configs/hu_turn2_stage9f_cse2_csemax2_bothseat_guarded_production_preset.json"
        ).read_text(encoding="utf-8")
    )

    assert config["status"] == "guarded_production_preset_ready_not_enabled"
    assert config["guarded_production_candidate"] is True
    assert config["requires_explicit_enable"] is True
    assert config["production_default"] is False
    assert config["production_p2_fixed"] is False
    assert config["validation_default"] is False
    assert config["full_replacement_enabled"] is False
    assert config["ai_profile"] == "stage9f_cse2_csemax2_bothseat"
    assert config["source_rollout_config"] == (
        "configs/hu_turn2_stage9f_cse2_csemax2_bothseat_limited_canary_rollout.json"
    )
    assert config["rollback"]["preset"] == "stage9f_off"
    assert config["rollback"]["config_file"] == "configs/hu_turn2_stage9f_canary_presets.json"

    assert config["model"]["path"] == (
        "models/hu_turn2_stage8b_safe_lcb196_20k_reference_override_cached_rank_wide.pt"
    )
    assert config["model"]["role"] == "candidate_generator"
    assert config["t3_continuation"]["profile"] == "stage7_m5_r10"
    assert config["t3_continuation"]["hu_turn3_min_margin"] == 5.0
    assert config["t3_continuation"]["hu_turn3_reference_min_margin"] == 10.0
    assert config["t3_continuation"]["full_replacement_enabled"] is False

    runtime = config["runtime"]
    assert runtime["enabled_when_explicitly_selected"] is True
    assert runtime["allowed_seats"] == ["first", "second"]
    assert runtime["full_replacement_enabled"] is False
    parsed = parse_topk_configs(runtime["config_string"])[0]
    assert parsed.top_k == 3
    assert parsed.mc_samples == 16
    assert parsed.confirm_mc_samples == 32
    assert parsed.confirm_se_multiplier == 2.0
    assert parsed.max_confirm_se == 2.0
    assert parsed.first_min_confirm_delta == 0.0
    assert parsed.second_min_confirm_delta == 0.0
    assert parsed.min_predicted_delta == 0.0
    assert parsed.allowed_seats == ("first", "second")

    assert config["safety"]["baseline_turn2_is_default_until_explicit_enable"] is True
    assert config["safety"]["stage9f_selective_override_only"] is True
    assert config["safety"]["runtime_logs_must_be_replay_ready"] is True
    assert config["primary_metric"]["source"] == "realized_fired_whole_game_delta"
    assert config["primary_metric"]["confirm_delta_performance_claim_allowed"] is False

    evidence = config["promotion_evidence"]
    monitoring = config["monitoring"]
    assert evidence["c4_realized_fires"] == 4000
    assert evidence["c4_per_fire_ci95_low"] > 0.0
    assert evidence["c4_non_fired_nonzero_count"] == 0
    assert evidence["production_like_canary_decisions"] == 60000
    assert evidence["production_like_canary_realized_fires"] >= 500
    assert evidence["production_like_canary_per_fire_ci95_low"] > 0.0
    assert evidence["production_like_canary_first_per_fire_ci95_low"] > 0.0
    assert evidence["production_like_canary_second_per_fire_ci95_low"] > 0.0
    assert evidence["production_like_canary_non_fired_nonzero_count"] == 0
    assert evidence["production_like_canary_replay_ready"] == 60000
    assert evidence["production_like_canary_p95_loss"] <= (
        monitoring["warning_thresholds"]["p95_fired_loss"]
    )
    assert evidence["production_like_canary_max_loss"] < (
        monitoring["rollback_thresholds"]["max_fired_loss"]
    )
    assert evidence["production_like_canary_latency_p95_ms"] > (
        monitoring["warning_thresholds"]["latency_p95_ms"]
    )
    assert evidence["production_like_canary_latency_p95_ms"] < (
        monitoring["rollback_thresholds"]["latency_p95_ms"]
    )
    assert evidence["production_like_canary_latency_warning"] is True

    assert monitoring["promotion_minimum_paired_decisions"] == 40000
    assert monitoring["promotion_minimum_fired_decisions"] == 500
    assert monitoring["required"]["missing_replay_fields"] == 0
    assert monitoring["required"]["non_fired_nonzero_count"] == 0
    assert monitoring["rollback_thresholds"]["per_fire_ci95_low"] == 0.0
    assert monitoring["rollback_thresholds"]["first_per_fire_ci95_low"] == 0.0
    assert monitoring["rollback_thresholds"]["second_per_fire_ci95_low"] == 0.0

    assert config["decision"]["guarded_production_preset"] == "Ready"
    assert config["decision"]["production_default"] == "No-Go until explicitly enabled"
    assert config["decision"]["p2_fixed"] == "No-Go until explicitly accepted"
    assert config["decision"]["teacher_50k"] == "No-Go"
    assert config["decision"]["turn1_training"] == "No-Go"
    assert config["decision"]["full_replacement"] == "No-Go"


def test_stage9f_guarded_preset_verifier_passes_current_production_like_audit():
    result = verify_guarded_preset(
        preset_path=Path(
            "configs/hu_turn2_stage9f_cse2_csemax2_bothseat_guarded_production_preset.json"
        ),
        audit_dir=Path(
            "outputs/evals/hu_turn2_stage9f_bothseat_profile_production_like_canary30000/audit"
        ),
    )

    assert result["status"] == "pass"
    assert result["failures"] == []
    assert result["decision"]["guarded_production_preset"] == "Ready"
    assert result["decision"]["production_default"] == "No-Go until explicitly enabled"
    assert result["decision"]["p2_fixed"] == "No-Go until explicitly accepted"
    assert result["metrics"]["decisions"] == 60000
    assert result["metrics"]["realized_fires"] >= 500
    assert result["metrics"]["per_fire_ci95_low"] > 0.0
    assert result["metrics"]["non_fired_nonzero_count"] == 0
    assert result["metrics"]["stage_a_latency_p95_ms"] > 0.0
    assert result["metrics"]["confirm_latency_p95_ms"] > 0.0
    assert result["metrics"]["overhead_latency_p95_ms"] < 50.0
    assert any("latency_p95_ms" in warning for warning in result["warnings"])
    assert any("stage_a=" in warning and "confirm=" in warning for warning in result["warnings"])


def test_stage9f_guarded_preset_verifier_accepts_guarded_canary_thresholds():
    result = verify_guarded_preset(
        preset_path=Path(
            "configs/hu_turn2_stage9f_cse2_csemax2_bothseat_guarded_production_preset.json"
        ),
        audit_dir=Path("outputs/evals/hu_turn2_stage9f_guarded_production_canary/audit"),
        minimum_decisions=20000,
        minimum_realized_fires=150,
        verification_scope="guarded_canary",
    )

    assert result["status"] == "pass"
    assert result["failures"] == []
    assert result["metrics"]["verification_scope"] == "guarded_canary"
    assert result["metrics"]["minimum_decisions"] == 20000
    assert result["metrics"]["minimum_realized_fires"] == 150
    assert result["metrics"]["decisions"] == 20000
    assert result["metrics"]["realized_fires"] >= 150
    assert result["metrics"]["per_fire_ci95_low"] > 0.0
    assert result["metrics"]["non_fired_nonzero_count"] == 0
    assert result["decision"]["guarded_production_preset"] == "Ready"
    assert result["decision"]["production_default"] == "No-Go until explicitly enabled"
    assert result["decision"]["p2_fixed"] == "No-Go until explicitly accepted"


def test_stage9f_guarded_preset_verifier_fails_on_negative_ci(tmp_path):
    source = Path(
        "outputs/evals/hu_turn2_stage9f_bothseat_profile_production_like_canary30000/audit"
    )
    audit_dir = tmp_path / "audit"
    shutil.copytree(source, audit_dir)

    decision_csv = audit_dir / "profile_canary_decision_summary.csv"
    lines = decision_csv.read_text(encoding="utf-8").splitlines()
    headers = lines[0].split(",")
    values = lines[1].split(",")
    values[headers.index("realized_per_fire_delta_ci95_low")] = "-0.01"
    decision_csv.write_text(",".join(headers) + "\n" + ",".join(values) + "\n", encoding="utf-8")

    result = verify_guarded_preset(
        preset_path=Path(
            "configs/hu_turn2_stage9f_cse2_csemax2_bothseat_guarded_production_preset.json"
        ),
        audit_dir=audit_dir,
    )

    assert result["status"] == "fail"
    assert result["decision"]["guarded_production_preset"] == "No-Go"
    assert any(failure.startswith("aggregate_per_fire_positive") for failure in result["failures"])


def test_stage9f_guarded_production_canary_plan_is_explicit_and_rollback_safe():
    plan = json.loads(
        Path("configs/hu_turn2_stage9f_guarded_production_canary_plan.json").read_text(
            encoding="utf-8"
        )
    )

    assert plan["status"] == "gcp_canary_pass_guarded_preset_ready"
    assert plan["production_default"] is False
    assert plan["production_p2_fixed"] is False
    assert plan["requires_explicit_enable"] is True
    assert plan["full_replacement_enabled"] is False
    assert plan["guarded_preset"] == (
        "configs/hu_turn2_stage9f_cse2_csemax2_bothseat_guarded_production_preset.json"
    )
    assert plan["rollback_preset"] == "stage9f_off"
    assert plan["profiles"] == {
        "candidate": "stage9f_cse2_csemax2_bothseat",
        "baseline": "stage7_m5_r10",
        "t3_continuation": "stage7_m5_r10",
    }

    runtime = plan["runtime"]
    assert runtime["selective_override_only"] is True
    parsed = parse_topk_configs(runtime["config_string"])[0]
    assert parsed.top_k == 3
    assert parsed.mc_samples == 16
    assert parsed.confirm_mc_samples == 32
    assert parsed.confirm_se_multiplier == 2.0
    assert parsed.max_confirm_se == 2.0
    assert parsed.allowed_seats == ("first", "second")

    gcp = plan["recommended_gcp_canary"]
    assert gcp["total_games"] == 10000
    assert gcp["shard_games"] == 1000
    assert gcp["vm_count"] == 10
    assert gcp["runner_script"] == "scripts/Start-GcpHuTurn2Stage9fProfileCanaryRun.ps1"
    assert gcp["receive_script"] == "scripts/Receive-GcpHuTurn2Stage9fProfileCanaryRun.ps1"
    assert gcp["verify_script"] == "scripts/Verify-HuTurn2Stage9fGuardedProductionCanary.ps1"

    dry_run = plan["dry_run_evidence"]
    assert dry_run["status"] == "pass"
    assert dry_run["execution"] == "dry_run"
    assert dry_run["create_instances"] is False
    assert dry_run["run_name"] == (
        "regular-hu-t2-stage9f-guarded-production-canary-dryrun-20260623-001"
    )
    assert "-DryRun" in dry_run["command"]
    assert "-CreateInstances" not in dry_run["command"]
    observed = dry_run["observed"]
    assert observed["profile_a"] == "stage9f_cse2_csemax2_bothseat"
    assert observed["profile_b"] == "stage7_m5_r10"
    assert observed["total_games"] == gcp["total_games"]
    assert observed["total_hands"] == 20000
    assert observed["total_decisions"] == 20000
    assert observed["total_shards"] == 10
    assert observed["vm_count"] == gcp["vm_count"]
    assert observed["machine_type"] == gcp["machine_type"]
    assert observed["seed_stride"] == gcp["seed_stride"]
    assert observed["shard_seeds"] == [
        2026069001,
        2027069001,
        2028069001,
        2029069001,
        2030069001,
        2031069001,
        2032069001,
        2033069001,
        2034069001,
        2035069001,
    ]
    assert dry_run["decision"]["actual_gcp_canary"] == (
        "ready_to_start_after_explicit_approval"
    )
    assert dry_run["decision"]["teacher_50k"] == "No-Go"
    assert dry_run["decision"]["turn1_training"] == "No-Go"

    active = plan["active_run"]
    assert active["status"] == "complete"
    assert active["run_name"] == "regular-hu-t2-stage9f-guarded-production-canary-20260623-001"
    assert active["profile_a"] == "stage9f_cse2_csemax2_bothseat"
    assert active["profile_b"] == "stage7_m5_r10"
    assert active["total_games"] == gcp["total_games"]
    assert active["total_decisions"] == 20000
    assert active["total_shards"] == 10
    assert active["vm_count"] == 10
    assert active["machine_type"] == "e2-highcpu-4"
    assert active["initial_status"]["running_instances"] == 10
    assert active["initial_status"]["failed_shards"] == 0
    assert active["initial_status"]["representative_serial_log"] == (
        "setup_complete_rust_direct_available_and_shard_loop_started"
    )
    assert active["spot_retries"]["restarted_shards"] == [5, 3, 4]
    assert active["spot_retries"]["final_completed_shards"] == 10
    assert active["spot_retries"]["final_missing_shards"] == 0
    assert active["spot_retries"]["final_failed_shards"] == 0
    assert active["spot_retries"]["final_running_instances"] == 0
    canary = active["guarded_canary_result"]
    assert canary["verification_status"] == "pass"
    assert canary["paired_seeds"] == 10000
    assert canary["decisions"] == 20000
    assert canary["realized_fires"] >= plan["go_thresholds"]["minimum_realized_fires"]
    assert canary["realized_per_fire_delta_ci95_low"] > 0.0
    assert canary["first_per_fire_delta_ci95_low"] > 0.0
    assert canary["second_per_fire_delta_ci95_low"] > 0.0
    assert canary["non_fired_nonzero_count"] == 0
    assert canary["replay_ready"] == 20000
    assert canary["missing_replay_field_counts"] == {}
    assert canary["p95_loss"] <= plan["go_thresholds"]["p95_fired_loss_max"]
    assert canary["max_loss"] < plan["go_thresholds"]["max_fired_loss_max"]
    assert canary["latency_p95_ms"] > plan["go_thresholds"]["latency_p95_ms_warning"]
    assert canary["latency_p95_ms"] < plan["go_thresholds"]["latency_p95_ms_rollback"]
    assert canary["latency_warning"] is True
    assert canary["latency_component_overhead_p95_ms"] < 50.0
    assert active["decision"]["guarded_production_preset"] == "Ready"
    assert active["decision"]["production_default"] == "No-Go until explicitly enabled"
    assert active["decision"]["production_p2_fixed"] == "No-Go until explicitly accepted"
    assert active["decision"]["teacher_50k"] == "No-Go"
    assert active["decision"]["turn1_training"] == "No-Go"

    commands = plan["recommended_commands"]
    assert "-DryRun" in commands["dry_run"]
    assert "-CreateInstances" in commands["start"]
    assert "stage9f_cse2_csemax2_bothseat" in commands["start"]
    assert "stage7_m5_r10" in commands["start"]
    assert "Verify-HuTurn2Stage9fGuardedProductionCanary.ps1" in commands["verify"]

    thresholds = plan["go_thresholds"]
    assert thresholds["minimum_decisions"] == 20000
    assert thresholds["minimum_realized_fires"] == 150
    assert thresholds["aggregate_per_fire_ci95_low_min"] == 0.0
    assert thresholds["first_per_fire_ci95_low_min"] == 0.0
    assert thresholds["second_per_fire_ci95_low_min"] == 0.0
    assert thresholds["non_fired_nonzero_count"] == 0
    assert thresholds["latency_p95_ms_rollback"] == 1500.0
    assert thresholds["latency_p95_ms_warning"] == 800.0

    assert "latency_p95_ms >= 1500" in plan["stop_or_rollback_if"]
    assert plan["decision_after_pass"]["teacher_50k"].startswith("still No-Go")
    assert plan["decision_after_pass"]["turn1_training"].startswith("still No-Go")


def test_stage9f_p2_acceptance_candidate_is_accepted_for_experiments_only():
    config = json.loads(
        Path("configs/hu_turn2_stage9f_p2_acceptance_candidate.json").read_text(
            encoding="utf-8"
        )
    )

    assert config["status"] == "p2_accepted_for_experiments_not_production_default"
    assert config["applies_runtime_changes"] is False
    assert config["fixed_p2_profile_available"] is True
    assert config["fixed_p2_profile"] == "stage9f_p2"
    assert config["requires_explicit_acceptance"] is False
    assert config["acceptance_scope"] == "fixed_t2_p2_for_post_acceptance_experiments_only"
    assert config["production_default"] is False
    assert config["production_p2_fixed"] is False
    assert config["full_replacement_enabled"] is False

    candidate = config["candidate"]
    assert candidate["ai_profile"] == "stage9f_cse2_csemax2_bothseat"
    assert candidate["fixed_p2_profile"] == "stage9f_p2"
    assert candidate["guarded_preset"] == (
        "configs/hu_turn2_stage9f_cse2_csemax2_bothseat_guarded_production_preset.json"
    )
    assert candidate["canary_plan"] == "configs/hu_turn2_stage9f_guarded_production_canary_plan.json"
    assert candidate["selective_override_only"] is True
    assert candidate["allowed_seats"] == ["first", "second"]
    assert candidate["t3_continuation"]["profile"] == "stage7_m5_r10"
    assert candidate["t3_continuation"]["hu_turn3_min_margin"] == 5.0
    assert candidate["t3_continuation"]["hu_turn3_reference_min_margin"] == 10.0
    parsed = parse_topk_configs(candidate["runtime"])[0]
    assert parsed.top_k == 3
    assert parsed.mc_samples == 16
    assert parsed.confirm_mc_samples == 32
    assert parsed.confirm_se_multiplier == 2.0
    assert parsed.max_confirm_se == 2.0
    assert parsed.allowed_seats == ("first", "second")

    evidence = config["acceptance_evidence"]
    assert evidence["c4_validation"]["realized_fires"] == 4000
    assert evidence["c4_validation"]["per_fire_ci95_low"] > 0.0
    assert evidence["c4_validation"]["non_fired_nonzero_count"] == 0
    assert evidence["production_like_canary30000"]["decisions"] == 60000
    assert evidence["production_like_canary30000"]["realized_fires"] >= 500
    assert evidence["production_like_canary30000"]["per_fire_ci95_low"] > 0.0
    assert evidence["production_like_canary30000"]["first_per_fire_ci95_low"] > 0.0
    assert evidence["production_like_canary30000"]["second_per_fire_ci95_low"] > 0.0
    assert evidence["production_like_canary30000"]["non_fired_nonzero_count"] == 0
    assert evidence["guarded_canary10000"]["verification_status"] == "pass"
    assert evidence["guarded_canary10000"]["decisions"] == 20000
    assert evidence["guarded_canary10000"]["realized_fires"] >= 150
    assert evidence["guarded_canary10000"]["per_fire_ci95_low"] > 0.0
    assert evidence["guarded_canary10000"]["first_per_fire_ci95_low"] > 0.0
    assert evidence["guarded_canary10000"]["second_per_fire_ci95_low"] > 0.0
    assert evidence["guarded_canary10000"]["non_fired_nonzero_count"] == 0
    assert evidence["guarded_canary10000"]["replay_ready"] == 20000
    assert evidence["guarded_canary10000"]["missing_replay_field_counts"] == {}
    assert evidence["guarded_canary10000"]["p95_loss"] <= (
        config["acceptance_criteria"]["p95_loss_max"]
    )
    assert evidence["guarded_canary10000"]["max_loss"] < (
        config["acceptance_criteria"]["max_loss_lt"]
    )
    assert evidence["guarded_canary10000"]["latency_p95_ms"] < (
        config["acceptance_criteria"]["latency_p95_rollback_ms"]
    )

    decision = config["acceptance_decision"]
    assert decision["p2_fixed_recommendation"] == "Accepted for post-acceptance experiments"
    assert decision["production_default_recommendation"].startswith("Eligible")
    assert decision["teacher_50k"].startswith("No-Go")
    assert decision["turn1_training"].startswith("Go for new post-acceptance T1 experiments")
    assert decision["full_replacement"] == "No-Go"
    assert decision["rollback"] == "stage9f_off"

    enable = config["enable_preview"]
    assert enable["description"].startswith("P2 accepted for experiments only")
    assert enable["config_file"] == "configs/hu_turn2_stage9f_canary_presets.json"
    assert enable["set_preset"] == "stage9f_cse2_csemax2_bothseat"
    assert enable["set_production_default"] is True
    assert enable["keep_full_replacement_enabled"] is False
    assert enable["rollback_preset"] == "stage9f_off"
