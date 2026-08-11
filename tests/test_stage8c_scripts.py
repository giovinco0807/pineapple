from pathlib import Path

from ofc_regular.evaluate_hu_turn2_stage8_seat_swap import ModelParts, _t3_policy_kwargs
from ofc_regular.evaluate_hu_turn2_stage8b_topk_mc_rerank import (
    HuTurn2Stage8bTopKMcRerankPolicy,
    TopKMcRerankConfig,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _dummy_model_parts() -> ModelParts:
    return ModelParts(
        opening="opening",
        turn1="turn1",
        turn2_baseline="turn2",
        turn3="turn3",
        hu_turn3_stage7="stage7",
        hu_turn3_reference="stage3_reference",
        hu_turn2_stage8="stage8",
    )


def test_first_seat_c4_runner_defaults_to_base_model_and_first_seat():
    script = (REPO_ROOT / "scripts" / "Run-HuTurn2Stage8cFirstSeatC4.ps1").read_text(encoding="utf-8")

    assert "hu_turn2_current_fl_ev_stage8c_mixed5k_mc16.pt" in script
    assert "seat=first" in script
    assert "seat_scope = \"first\"" in script
    assert "production_p2_fixed = \"No-Go\"" in script
    assert "AggregateOutputDir" in script
    assert "analyze_hu_turn2_stage8c_topk_per_fire" in script
    assert "$runnerParams = @{" in script
    assert "& $runner @runnerParams" in script
    assert "AllowConcurrentPython" in script
    assert "Concurrent CUDA Python job detected" in script
    assert "WaitForGpuClear" in script
    assert "GpuWaitTimeoutSeconds" in script
    assert "GpuPollSeconds" in script
    assert '$usesGpu = $Device -ne "cpu"' in script
    assert "[string]$T3Continuation = \"stage3_reference_default\"" in script
    assert "-T3Continuation" in script
    assert "t3_continuation = $T3Continuation" in script


def test_first_seat_c4_runner_rejects_non_first_seat_configs():
    script = (REPO_ROOT / "scripts" / "Run-HuTurn2Stage8cFirstSeatC4.ps1").read_text(encoding="utf-8")

    assert "First-seat C4 runner only accepts configs with seat=first" in script
    assert "First-seat C4 runner rejects non-first seat configs" in script
    assert "seat=(second|both|all|\\*)" in script
    assert "DryRun" in script
    assert "Aggregate output requires runtime_decisions.jsonl" in script
    assert "use -NoAggregate when -NoDecisionLog is set" in script
    assert "concurrent_gpu_python_process_count" in script
    assert "ofc_regular|ai\\.tutor|ofc-pineapple|regular-ofc-pineapple" in script
    assert "Waiting for concurrent CUDA Python job(s) to finish" in script
    assert "Re-run with -WaitForGpuClear to wait" in script
    assert "$usesGpu -and -not $AllowConcurrentPython" in script


def test_gcp_stage8c_topk_runner_is_first_seat_and_aggregate_ready():
    start_script = (REPO_ROOT / "scripts" / "Start-GcpHuTurn2Stage8cTopkPerFireRun.ps1").read_text(
        encoding="utf-8"
    )
    receive_script = (
        REPO_ROOT / "scripts" / "Receive-GcpHuTurn2Stage8cTopkPerFireRun.ps1"
    ).read_text(encoding="utf-8")
    status_script = (
        REPO_ROOT / "scripts" / "Get-GcpHuTurn2Stage8cTopkPerFireRunStatus.ps1"
    ).read_text(encoding="utf-8")

    assert "hu_t2_stage8c_topk_per_fire" in start_script
    assert "hu_t2_stage8c_topk_risk_data" in start_script
    assert "seat=first" in start_script
    assert "AllowRiskDataSeatScope" in start_script
    assert "AllowPositionSpecificSeatScope" in start_script
    assert "position_specific_seat_scope" in start_script
    assert "POSITION_SPECIFIC_SEAT_SCOPE" in start_script
    assert "fcd/scd guard" in start_script
    assert "first_min_confirm_delta" in start_script
    assert "second_min_confirm_delta" in start_script
    assert "requires -AllowRiskDataSeatScope for second-seat risk data configs" in start_script
    assert "-AllowPositionSpecificSeatScope with fcd/scd guard" in start_script
    assert "rejects broad/ambiguous seat configs" in start_script
    assert "evaluate_hu_turn2_stage8b_topk_mc_rerank" in start_script
    assert "hu_turn2_current_fl_ev_stage8c_mixed5k_mc16.pt" in start_script
    assert "cargo build --release" in start_script
    assert "rust_direct_available" in start_script
    assert "--target-realized-overrides-per-seed" in start_script
    assert "--target-risk-vetoes-per-seed" in start_script
    assert "TargetRiskVetoesPerSeed" in start_script
    assert "target_risk_vetoes_per_seed" in start_script
    assert "$workerStride = if ($StartShardIndices.Count -gt 0) { $totalShards } else { $VmCount }" in start_script
    assert "$dryRunWorkerStride = if ($StartShardIndices.Count -gt 0) { $dryRunTotalShards } else { $VmCount }" in start_script
    assert "worker_stride" in start_script
    assert "start_shards" in start_script
    assert '"VM_COUNT=$workerStride"' in start_script
    assert "Stage8cRiskAuditOnly" in start_script
    assert "--stage8c-risk-audit-only" in start_script
    assert "STAGE8C_RISK_AUDIT_ONLY" in start_script
    assert "--t3-continuation \"$T3_CONTINUATION\"" in start_script
    assert '"t3_continuation":"$T3_CONTINUATION"' in start_script
    assert "T3_CONTINUATION=$T3Continuation" in start_script
    assert "Stage8cRiskModel" in start_script
    assert "Stage8cRiskThreshold" in start_script
    assert "Stage8cRiskRankMin" in start_script
    assert "Stage8cRiskRankMax" in start_script
    assert "STAGE8C_RISK_MODEL" in start_script
    assert "--stage8c-risk-model" in start_script
    assert "--stage8c-risk-threshold" in start_script
    assert "--stage8c-risk-rank-min" in start_script
    assert "--stage8c-risk-rank-max" in start_script
    assert '"stage8c_risk_model":"$STAGE8C_RISK_MODEL"' in start_script
    assert "Stage8cFireSelectorDirectFire" in start_script
    assert "STAGE8C_FIRE_SELECTOR_DIRECT_FIRE=\"$(meta STAGE8C_FIRE_SELECTOR_DIRECT_FIRE)\"" in start_script
    assert "STAGE8C_FIRE_SELECTOR_DIRECT_FIRE=$([int][bool]$Stage8cFireSelectorDirectFire)" in start_script
    assert "--stage8c-fire-selector-direct-fire" in start_script
    assert '"stage8c_fire_selector_direct_fire":"$STAGE8C_FIRE_SELECTOR_DIRECT_FIRE"' in start_script
    assert "RISK_DATA_COLLECTION" in start_script
    assert "PHASE_NAME" in start_script
    assert "risk_data_collection" in start_script
    assert "DryRun" in start_script
    assert "execution = \"dry_run\"" in start_script
    assert "--write-decision-log" in start_script
    assert "CreateInstances" in start_script
    assert "SPOT" in start_script
    assert '"status":"running"' in start_script

    assert "analyze_hu_turn2_stage8c_topk_per_fire" in receive_script
    assert "Missing Stage8c TopK result shards" in receive_script
    assert "AllowPartial" in receive_script
    assert "Aggregating partial Stage8c TopK results" in receive_script
    assert "No completed Stage8c TopK result shards found" in receive_script
    assert "completed_shards" in receive_script
    assert "missing_shards" in receive_script
    assert "shards_manifest.jsonl" in receive_script
    assert "running_instances" in status_script
    assert "running_shards" in status_script
    assert "completed_shards" in status_script


def test_gcp_stage9f_profile_canary_passes_seed_stride_to_matchup_runner():
    script = (REPO_ROOT / "scripts" / "Start-GcpHuTurn2Stage9fProfileCanaryRun.ps1").read_text(
        encoding="utf-8"
    )

    assert 'SEED_STRIDE="$(meta SEED_STRIDE)"' in script
    assert '--seed-stride "$SEED_STRIDE"' in script
    assert '"SEED_STRIDE=$SeedStride"' in script
    assert 'if ($SeedStride -le 0) { throw "SeedStride must be positive" }' in script
    assert "$seed = $BaseSeed + ($shardIndex * $ShardGames * $SeedStride)" in script
    assert "seed_stride = $SeedStride" in script
    assert script.count("base_seed = $BaseSeed") >= 2
    assert script.count("hu_turn1_topk_config = $HuTurn1TopkConfig") >= 2
    assert script.count("hu_turn1_safe_selector_model = $HuTurn1SafeSelectorModel") >= 2
    assert script.count("hu_turn2_stage8b_model = $HuTurn2Stage8bModel") >= 2


def test_gcp_stage9f_profile_canary_retries_spot_stockouts_in_other_zones():
    script = (REPO_ROOT / "scripts" / "Start-GcpHuTurn2Stage9fProfileCanaryRun.ps1").read_text(
        encoding="utf-8"
    )

    assert "$zoneCandidates" in script
    assert "trying the next configured zone" in script
    assert "Failed to create instance $vmName in all configured zones" in script
    assert "[switch]$ReuseRunArtifacts" in script
    assert "Cannot reuse missing run artifact" in script
    assert script.count("reuse_run_artifacts = [bool]$ReuseRunArtifacts") >= 2


def test_stage8c_topk_local_runner_makes_t3_continuation_explicit():
    script = (REPO_ROOT / "scripts" / "Run-HuTurn2Stage8cTopkPerFireEval.ps1").read_text(
        encoding="utf-8"
    )

    assert "[string]$T3Continuation = \"stage3_reference_default\"" in script
    assert "--t3-continuation" in script
    assert "$T3Continuation" in script
    assert "Stage8cRiskModel" in script
    assert "Stage8cRiskThreshold" in script
    assert "Stage8cRiskRankMin" in script
    assert "Stage8cRiskRankMax" in script
    assert "TargetRiskVetoesPerSeed" in script
    assert "Stage8cRiskAuditOnly" in script
    assert "--target-risk-vetoes-per-seed" in script
    assert "--stage8c-risk-audit-only" in script
    assert "--stage8c-risk-model" in script
    assert "--stage8c-risk-threshold" in script
    assert "--stage8c-risk-rank-min" in script
    assert "--stage8c-risk-rank-max" in script


def test_stage8c_replay_ready_postprocess_runs_full_loss_target_pipeline():
    script = (REPO_ROOT / "scripts" / "Run-HuTurn2Stage8cReplayReadyPostprocess.ps1").read_text(
        encoding="utf-8"
    )

    assert "analyze_hu_turn2_stage8c_topk_per_fire" in script
    assert "prepare_hu_turn2_stage8b_topk_hard_negatives" in script
    assert "replay_hu_turn2_stage8b_topk_hard_negatives" in script
    assert "prepare_hu_turn2_stage8b_counterfactual_loss_targets" in script
    assert "collect_hu_turn2_stage8b_loss_targets" in script
    assert "analyze_hu_turn2_stage8b_risk_targets" in script
    assert "analyze_hu_turn2_stage8c_risk_target_gap" in script
    assert "--include-non-loss-controls" in script
    assert "topk_all_fired_deduped.jsonl" in script
    assert "topk_false_positive_hard_negatives.jsonl" in script
    assert "topk_hard_negative_replay_summary.csv" in script
    assert "risk_target_audit" in script
    assert "risk_target_gap" in script
    assert "[string]$T3Continuation = \"stage3_reference_default\"" in script
    assert "stage7_m5_r10" in script
    assert "SkipAllFiredReplay" in script
    assert "SkipFalsePositiveReplay" in script
    assert "SkipLossTargetCollection" in script
    assert "SkipRiskTargetAudit" in script
    assert "SkipRiskTargetGap" in script
    assert "DryRun" in script
    assert "AllFiredReplayOffset" in script
    assert "AllFiredReplayLimit" in script
    assert "FalsePositiveReplayOffset" in script
    assert "FalsePositiveReplayLimit" in script
    assert "--offset" in script
    assert "--limit" in script
    assert "production_p2_fixed = \"No-Go\"" in script
    assert "teacher_50k = \"No-Go\"" in script


def test_stage8c_replay_chunk_runner_supports_offset_limit_and_merge():
    script = (REPO_ROOT / "scripts" / "Run-HuTurn2Stage8cReplayChunks.ps1").read_text(
        encoding="utf-8"
    )

    assert "InputJsonl" in script
    assert "ChunkSize" in script
    assert "StartOffset" in script
    assert "TotalRows" in script
    assert "--offset" in script
    assert "--limit" in script
    assert "topk_hard_negative_replay_summary.csv" in script
    assert "topk_hard_negative_replay_teacher.jsonl" in script
    assert "merged_manifest.json" in script
    assert "SkipExisting" in script
    assert "NoMerge" in script
    assert "DryRun" in script
    assert "stage3_reference_default" in script
    assert "stage7_m5_r10" in script
    assert "production_p2_fixed = \"No-Go\"" in script
    assert "teacher_50k = \"No-Go\"" in script


def test_gcp_stage8c_replay_chunk_runner_has_status_receive_and_loss_target_flow():
    start_script = (REPO_ROOT / "scripts" / "Start-GcpHuTurn2Stage8cReplayChunksRun.ps1").read_text(
        encoding="utf-8"
    )
    status_script = (REPO_ROOT / "scripts" / "Get-GcpHuTurn2Stage8cReplayChunksRunStatus.ps1").read_text(
        encoding="utf-8"
    )
    receive_script = (REPO_ROOT / "scripts" / "Receive-GcpHuTurn2Stage8cReplayChunksRun.ps1").read_text(
        encoding="utf-8"
    )

    assert "hu_t2_stage8c_replay_chunks" in start_script
    assert "topk_all_fired_deduped.jsonl" in start_script
    assert "replay_hu_turn2_stage8b_topk_hard_negatives" in start_script
    assert "--offset" in start_script
    assert "--limit" in start_script
    assert "CreateInstances" in start_script
    assert "SPOT" in start_script
    assert "DryRun" in start_script
    assert "stage3_reference_default" in start_script
    assert "stage7_m5_r10" in start_script
    assert "production_training = \"No-Go\"" in start_script
    assert "fifty_k_teacher = \"No-Go\"" in start_script

    assert "completed_shards" in status_script
    assert "missing_shards" in status_script
    assert "running_instances" in status_script

    assert "topk_hard_negative_replay_summary.csv" in receive_script
    assert "topk_hard_negative_replay_teacher.jsonl" in receive_script
    assert "prepare_hu_turn2_stage8b_counterfactual_loss_targets" in receive_script
    assert "collect_hu_turn2_stage8b_loss_targets" in receive_script
    assert "analyze_hu_turn2_stage8b_risk_targets" in receive_script
    assert "analyze_hu_turn2_stage8c_risk_target_gap" in receive_script
    assert "AllowPartial" in receive_script
    assert "SkipLossTargets" in receive_script
    assert "SkipRiskTargetAudit" in receive_script
    assert "SkipRiskTargetGap" in receive_script
    assert "risk_target_audit" in receive_script
    assert "risk_target_gap" in receive_script
    assert "production_p2_fixed = \"No-Go\"" in receive_script
    assert "teacher_50k = \"No-Go\"" in receive_script


def test_stage8_c3_seat_swap_runners_make_t3_continuation_explicit():
    local_script = (REPO_ROOT / "scripts" / "Run-HuTurn2Stage8C3LargerSeatSwap.ps1").read_text(
        encoding="utf-8"
    )
    gcp_script = (REPO_ROOT / "scripts" / "Start-GcpHuTurn2Stage8C3SeatSwapRun.ps1").read_text(
        encoding="utf-8"
    )

    for script in (local_script, gcp_script):
        assert "stage3_reference_default" in script
        assert "stage7_m5_r10" in script
        assert "--t3-continuation" in script

    assert "[string]$T3Continuation = \"stage3_reference_default\"" in local_script
    assert "--t3-continuation\", \"$T3Continuation\"" in local_script
    assert "[string]$T3Continuation = \"stage3_reference_default\"" in gcp_script
    assert "T3_CONTINUATION" in gcp_script
    assert "--t3-continuation \"$T3_CONTINUATION\"" in gcp_script
    assert '"t3_continuation":"$T3_CONTINUATION"' in gcp_script


def test_stage8_c4_high_mc_gcp_runner_makes_t3_continuation_explicit():
    script = (REPO_ROOT / "scripts" / "Start-GcpHuTurn2Stage8C4HighMcRun.ps1").read_text(
        encoding="utf-8"
    )

    assert "[string]$T3Continuation = \"stage3_reference_default\"" in script
    assert "stage7_m5_r10" in script
    assert "T3_CONTINUATION" in script
    assert "--t3-continuation \"$T3_CONTINUATION\"" in script
    assert '"t3_continuation":"$T3_CONTINUATION"' in script
    assert "t3_continuation = $T3Continuation" in script


def test_hu_turn2_teacher_scripts_make_t3_continuation_explicit():
    shard_script = (REPO_ROOT / "scripts" / "Run-HuTurn2TeacherShard.ps1").read_text(
        encoding="utf-8"
    )
    parallel_script = (
        REPO_ROOT / "scripts" / "Run-HuTurn2TeacherChunksParallel.ps1"
    ).read_text(encoding="utf-8")
    gcp_script = (REPO_ROOT / "scripts" / "Start-GcpHuTurn2PilotSpotRun.ps1").read_text(
        encoding="utf-8"
    )

    for script in (shard_script, parallel_script, gcp_script):
        assert "stage3_reference_default" in script
        assert "stage7_m5_r10" in script

    assert "[string]$T3Continuation = \"stage3_reference_default\"" in shard_script
    assert "--t3-continuation" in shard_script
    assert "$T3Continuation" in parallel_script
    assert "-T3Continuation $t3Continuation" in parallel_script
    assert "T3_CONTINUATION" in gcp_script
    assert "--t3-continuation \"$T3_CONTINUATION\"" in gcp_script
    assert '"t3_continuation":"$T3_CONTINUATION"' in gcp_script


def test_hu_turn2_eval_t3_continuation_defaults_to_stage3_reference():
    parts = _dummy_model_parts()

    kwargs = _t3_policy_kwargs(parts, "stage3_reference_default")

    assert kwargs["hu_turn3_stage7_enabled"] is False
    assert kwargs["hu_turn3_model"] is None
    assert kwargs["hu_turn3_reference_model"] == "stage3_reference"
    assert kwargs["hu_turn3_min_margin"] == 0.0
    assert kwargs["hu_turn3_reference_min_margin"] == 0.0


def test_hu_turn2_eval_t3_continuation_stage7_opt_in_uses_m5_r10():
    parts = _dummy_model_parts()

    kwargs = _t3_policy_kwargs(parts, "stage7_m5_r10")

    assert kwargs["hu_turn3_stage7_enabled"] is True
    assert kwargs["hu_turn3_model"] == "stage7"
    assert kwargs["hu_turn3_reference_model"] == "stage3_reference"
    assert kwargs["hu_turn3_min_margin"] == 5.0
    assert kwargs["hu_turn3_reference_min_margin"] == 10.0


def test_hu_turn2_topk_batched_continuation_config_tracks_t3_mode():
    config = TopKMcRerankConfig(top_k=3, mc_samples=1, min_delta=0.0)

    stage3_policy = HuTurn2Stage8bTopKMcRerankPolicy(
        hu_turn2_stage8b_model=None,
        topk_rerank_config=config,
        t3_continuation="stage3_reference_default",
    )
    stage7_policy = HuTurn2Stage8bTopKMcRerankPolicy(
        hu_turn2_stage8b_model=None,
        topk_rerank_config=config,
        t3_continuation="stage7_m5_r10",
    )

    assert stage3_policy.t3_continuation == "stage3_reference_default"
    assert stage3_policy._batched_config.stage7_enabled is False
    assert stage3_policy._batched_config.hu_turn3_reference_min_margin == 0.0

    assert stage7_policy.t3_continuation == "stage7_m5_r10"
    assert stage7_policy._batched_config.stage7_enabled is True
    assert stage7_policy._batched_config.hu_turn3_min_margin == 5.0
    assert stage7_policy._batched_config.hu_turn3_reference_min_margin == 10.0
