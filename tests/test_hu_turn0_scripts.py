from pathlib import Path


def test_turn0_gcp_wrapper_locks_fixed_continuations_and_models():
    text = Path("scripts/Start-GcpHuTurn0PilotRun.ps1").read_text(encoding="utf-8")

    assert 'TeacherModule = "ofc_regular.hu_turn0_teacher_pilot"' in text
    assert 'PhaseName = "hu_t0_stage1_pilot"' in text
    assert 'Profile = "stage18_p1"' in text
    assert 'OpponentProfile = "stage18_p1"' in text
    assert "hu_turn1_stage18_first1801_mc32_hgb_abs_aug3.pkl" in text
    assert "hu_turn1_stage18_highmc_safe_selector_a_meta_only.pkl" in text
    assert '[int]$FutureSamples = 4' in text
    assert '[int]$MaxActions = 0' in text
    assert '[string]$MachineType = "c4-highmem-2"' in text
    assert '[int]$RecordSkipBase = 0' in text
    assert 'RecordSkipBase = $RecordSkipBase' in text
    assert 'FastSkipRecords = [bool]$FastSkipRecords' in text
    assert '[string[]]$StartShards = @()' in text
    assert 'StartShards = $StartShards' in text
    assert '[string]$CandidateModel = ""' in text
    assert 'CandidateModel = $CandidateModel' in text
    assert 'CandidateTopk = $CandidateTopk' in text
    assert 'CandidateUnionMode = $CandidateUnionMode' in text
    assert '[int]$CpuThreads = 1' in text
    assert 'CpuThreads = $CpuThreads' in text


def test_generic_gcp_runner_uses_requested_teacher_module_and_phase():
    text = Path("scripts/Start-GcpHuTurn1PilotRun.ps1").read_text(encoding="utf-8")

    assert '[string]$TeacherModule = "ofc_regular.hu_turn1_teacher_pilot"' in text
    assert '[string]$PhaseName = "hu_t1_stage1_pilot"' in text
    assert 'TEACHER_MODULE="$(meta TEACHER_MODULE)"' in text
    assert 'PHASE_NAME="$(meta PHASE_NAME)"' in text
    assert '-B -m "$TEACHER_MODULE"' in text
    assert '"TEACHER_MODULE=$TeacherModule"' in text
    assert '"PHASE_NAME=$PhaseName"' in text
    assert '"CPU_THREADS=$CpuThreads"' in text
    assert 'export OMP_NUM_THREADS="$CPU_THREADS"' in text


def test_generic_gcp_runner_can_package_replay_inputs():
    text = Path("scripts/Start-GcpHuTurn1PilotRun.ps1").read_text(encoding="utf-8")

    assert "AdditionalPackageFiles" in text
    assert 'package_path = ("inputs/{0}" -f $destinationName)' in text
    assert "Additional package file names must be unique" in text


def test_turn0_receive_wrapper_selects_turn0_aggregator():
    text = Path("scripts/Receive-GcpHuTurn0PilotRun.ps1").read_text(encoding="utf-8")

    assert 'AggregatorModule = "ofc_regular.aggregate_hu_turn0_pilot"' in text
    assert 'MergedOutputName = "hu_turn0_stage1_pilot.jsonl"' in text
    assert 'ShardSummariesName = "hu_turn0_stage1_pilot_summaries.jsonl"' in text


def test_turn0_candidate_training_script_uses_state_holdout_and_union_coverage():
    text = Path("scripts/Train-HuTurn0Stage1Candidates.ps1").read_text(encoding="utf-8")

    assert '"--holdout-output", $holdoutPath' in text
    assert '"--validation-input", $ValidationInput' in text
    assert '"--source-weight", $sourceWeight' in text
    assert '"hgb_regressor"' in text
    assert '"extra_trees_regressor"' in text
    assert "train_hu_turn1_listwise_torch" in text
    assert '"--candidate-models"' not in text  # PowerShell continuation uses the native flag.
    assert "--candidate-models $trainedModels" in text
    assert "--candidate-model $OpeningModel" in text
    assert "--candidate-models $combinedModels" in text
    assert 'decision = "candidate_generator_only_not_p0"' in text
    assert "[switch]$SkipExtraTrees" in text


def test_turn0_stage2_training_pipeline_uses_mc4_as_explicit_validation():
    text = Path("scripts/Run-HuTurn0Stage2CandidateTraining.ps1").read_text(
        encoding="utf-8"
    )

    assert "prepare_hu_turn0_candidate_dataset" in text
    assert '-ValidationInput $mc4Validation' in text
    assert '-BroadValidationInput $mc1Validation' in text
    assert 'hu_turn0_terminal_rollout_mc1={0}' in text
    assert 'hu_turn0_terminal_rollout_mc4={0}' in text
    assert 'decision = "candidate_generator_comparison_only_not_p0"' in text


def test_turn0_stage3_high_mc_pipeline_trains_baseline_delta_and_calibrates_gate():
    text = Path("scripts/Run-HuTurn0Stage3HighMcTraining.ps1").read_text(
        encoding="utf-8"
    )

    assert "prepare_hu_turn0_high_mc_dataset" in text
    assert "-RegressionTarget baseline_delta" in text
    assert "analyze_hu_turn0_high_mc_gate" in text
    assert "hu_turn0_terminal_rollout_mc32=1.0" in text
    assert "--delta-se-weight-floor" in text
    assert "hgb_baseline-delta_sew" in text
    assert "hgb_baseline_delta_unaugmented_metrics.json" in text
    assert '"--suit-augmentations", "0"' in text
    assert '"--classification-target", "safe_lcb196"' in text
    assert "hgb_safe_lcb196" in text
    assert "-SkipExtraTrees:$SkipExtraTrees" in text
    assert 'decision = "calibration_only_fresh_seat_swap_required"' in text


def test_counterfactual_gcp_scripts_use_realized_events_and_nonoverlapping_seeds():
    start = Path("scripts/Start-GcpHuTurn0CounterfactualRun.ps1").read_text(
        encoding="utf-8"
    )
    receive = Path("scripts/Receive-GcpHuTurn0CounterfactualRun.ps1").read_text(
        encoding="utf-8"
    )
    status = Path("scripts/Get-GcpHuTurn0CounterfactualRunStatus.ps1").read_text(
        encoding="utf-8"
    )

    assert "evaluate_hu_turn0_counterfactual" in start
    assert "$startIndex * $SeedStride" in start
    assert "--events-output" in start
    assert "--config-id" in start
    assert "SafeSelectorModel" in start
    assert "--safe-selector-model" in start
    assert "SAFE_SELECTOR_THRESHOLD_FIRST" in start
    assert "aggregate_hu_turn0_counterfactual" in receive
    assert "--expected-paired-seeds-per-config" in receive
    assert "$completed.Count -eq 0" in receive
    assert "results/*/DONE" in status


def test_stage19_safe_selector_training_runner_uses_mc512_overrides_and_first_seat():
    text = Path("scripts/Run-HuTurn0Stage19SafeSelectorTraining.ps1").read_text(
        encoding="utf-8"
    )

    assert "prepare_hu_turn0_safe_selector_dataset" in text
    assert "selector_high_mc_only.jsonl" in text
    assert "selector_broad_with_mc512_overrides.jsonl" in text
    assert "--allowed-seats first" in text
    assert "--high-mc-weight" in text
    assert "select_hu_turn0_safe_selector_candidate" in text
    assert "not_approved_until_fixed_fresh_whole_game_holdout" in text


def test_stage19_selector_holdout_wrapper_preregisters_one_fresh_candidate():
    text = Path("scripts/Start-GcpHuTurn0Stage19SelectorHoldout.ps1").read_text(
        encoding="utf-8"
    )

    assert "preregistered_holdout.json" in text
    assert 'BaseSeed = 2026907001' in text
    assert 'stage19_safe_selector/0.5/999/first' in text
    assert "SafeSelectorThresholdFirst" in text
    assert "no_threshold_adaptation_after_holdout" in text
    assert "fresh holdout will not be launched" in text
