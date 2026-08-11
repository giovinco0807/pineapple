param(
    [Parameter(Mandatory = $true)]
    [string[]]$InputDirs,
    [string]$OutputRoot = "outputs\evals\hu_turn2_stage8c_replay_ready_postprocess",
    [int]$AllFiredFutureSamples = 128,
    [int]$FalsePositiveFutureSamples = 512,
    [int]$AllFiredReplayOffset = 0,
    [int]$AllFiredReplayLimit = 0,
    [int]$FalsePositiveReplayOffset = 0,
    [int]$FalsePositiveReplayLimit = 0,
    [int]$ReplaySeed = 2026064101,
    [int]$PredictionThreads = 1,
    [int]$OpeningLookaheadSamples = 32,
    [ValidateSet("stage3_reference_default", "stage7_m5_r10")]
    [string]$T3Continuation = "stage3_reference_default",
    [switch]$SkipAllFiredReplay,
    [switch]$SkipFalsePositiveReplay,
    [switch]$SkipLossTargetCollection,
    [switch]$SkipRiskTargetAudit,
    [switch]$SkipRiskTargetGap,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

if (-not $InputDirs -or $InputDirs.Count -eq 0) { throw "At least one InputDirs value is required" }
if ($AllFiredFutureSamples -le 0) { throw "AllFiredFutureSamples must be positive" }
if ($FalsePositiveFutureSamples -le 0) { throw "FalsePositiveFutureSamples must be positive" }
if ($AllFiredReplayOffset -lt 0) { throw "AllFiredReplayOffset must be non-negative" }
if ($AllFiredReplayLimit -lt 0) { throw "AllFiredReplayLimit must be non-negative" }
if ($FalsePositiveReplayOffset -lt 0) { throw "FalsePositiveReplayOffset must be non-negative" }
if ($FalsePositiveReplayLimit -lt 0) { throw "FalsePositiveReplayLimit must be non-negative" }
if ($PredictionThreads -lt 0) { throw "PredictionThreads must be non-negative" }
if ($OpeningLookaheadSamples -lt 0) { throw "OpeningLookaheadSamples must be non-negative" }

$resolvedInputDirs = @()
$decisionLogs = @()
foreach ($inputDir in $InputDirs) {
    if (-not (Test-Path -LiteralPath $inputDir)) { throw "InputDir not found: $inputDir" }
    $resolved = (Resolve-Path -LiteralPath $inputDir).Path
    $decisionLog = Join-Path $resolved "runtime_decisions.jsonl"
    if (-not (Test-Path -LiteralPath $decisionLog)) {
        throw "InputDir is missing runtime_decisions.jsonl: $inputDir"
    }
    $resolvedInputDirs += $resolved
    $decisionLogs += $decisionLog
}

$aggregateOutputDir = Join-Path $OutputRoot "aggregate"
$replayPackOutputDir = Join-Path $OutputRoot "replay_pack"
$allFiredReplayOutputDir = Join-Path $OutputRoot ("all_fired_replay_mc{0}" -f $AllFiredFutureSamples)
$falsePositiveReplayOutputDir = Join-Path $OutputRoot ("false_positive_replay_mc{0}" -f $FalsePositiveFutureSamples)
$lossTargetsAllFiredOutputDir = Join-Path $OutputRoot ("loss_targets_all_fired_mc{0}" -f $AllFiredFutureSamples)
$lossTargetsFalsePositiveOutputDir = Join-Path $OutputRoot ("loss_targets_false_positive_mc{0}" -f $FalsePositiveFutureSamples)
$lossTargetCollectionOutputDir = Join-Path $OutputRoot "loss_target_collection"
$riskTargetAuditOutputDir = Join-Path $OutputRoot "risk_target_audit"
$riskTargetGapOutputDir = Join-Path $OutputRoot "risk_target_gap"

$steps = [ordered]@{
    aggregate = $aggregateOutputDir
    replay_pack = $replayPackOutputDir
    all_fired_replay = if ($SkipAllFiredReplay) { "" } else { $allFiredReplayOutputDir }
    false_positive_replay = if ($SkipFalsePositiveReplay) { "" } else { $falsePositiveReplayOutputDir }
    loss_targets_all_fired = if ($SkipAllFiredReplay) { "" } else { $lossTargetsAllFiredOutputDir }
    loss_targets_false_positive = if ($SkipFalsePositiveReplay) { "" } else { $lossTargetsFalsePositiveOutputDir }
    loss_target_collection = if ($SkipLossTargetCollection) { "" } else { $lossTargetCollectionOutputDir }
    risk_target_audit = if ($SkipLossTargetCollection -or $SkipRiskTargetAudit) { "" } else { $riskTargetAuditOutputDir }
    risk_target_gap = if ($SkipLossTargetCollection -or $SkipRiskTargetGap) { "" } else { $riskTargetGapOutputDir }
}

if ($DryRun) {
    [pscustomobject]@{
        execution = "dry_run"
        input_dirs = $resolvedInputDirs
        decision_logs = $decisionLogs
        output_root = $OutputRoot
        steps = $steps
        all_fired_future_samples = $AllFiredFutureSamples
        false_positive_future_samples = $FalsePositiveFutureSamples
        all_fired_replay_offset = $AllFiredReplayOffset
        all_fired_replay_limit = $AllFiredReplayLimit
        false_positive_replay_offset = $FalsePositiveReplayOffset
        false_positive_replay_limit = $FalsePositiveReplayLimit
        replay_seed = $ReplaySeed
        prediction_threads = $PredictionThreads
        opening_lookahead_samples = $OpeningLookaheadSamples
        t3_continuation = $T3Continuation
        production_p2_fixed = "No-Go"
        t1_training = "No-Go"
        teacher_50k = "No-Go"
    } | ConvertTo-Json -Depth 6
    exit 0
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$aggregateArgs = @("-m", "ofc_regular.analyze_hu_turn2_stage8c_topk_per_fire")
foreach ($inputDir in $resolvedInputDirs) {
    $aggregateArgs += @("--input-dir", $inputDir)
}
$aggregateArgs += @("--output-dir", $aggregateOutputDir)
python @aggregateArgs
if ($LASTEXITCODE -ne 0) { throw "Aggregate failed with exit code $LASTEXITCODE" }

$packArgs = @("-m", "ofc_regular.prepare_hu_turn2_stage8b_topk_hard_negatives")
foreach ($decisionLog in $decisionLogs) {
    $packArgs += @("--decision-log", $decisionLog)
}
$packArgs += @("--output-dir", $replayPackOutputDir)
python @packArgs
if ($LASTEXITCODE -ne 0) { throw "Replay pack creation failed with exit code $LASTEXITCODE" }

$lossTargetDirs = @()
if (-not $SkipAllFiredReplay) {
    $replayArgs = @(
        "-m", "ofc_regular.replay_hu_turn2_stage8b_topk_hard_negatives",
        "--input-jsonl", (Join-Path $replayPackOutputDir "topk_all_fired_deduped.jsonl"),
        "--future-samples", "$AllFiredFutureSamples",
        "--seed", "$ReplaySeed",
        "--offset", "$AllFiredReplayOffset",
        "--prediction-threads", "$PredictionThreads",
        "--opening-lookahead-samples", "$OpeningLookaheadSamples",
        "--t3-continuation", "$T3Continuation",
        "--output-dir", "$allFiredReplayOutputDir"
    )
    if ($AllFiredReplayLimit -gt 0) { $replayArgs += @("--limit", "$AllFiredReplayLimit") }
    python @replayArgs
    if ($LASTEXITCODE -ne 0) { throw "All-fired replay failed with exit code $LASTEXITCODE" }

    $lossArgs = @("-m", "ofc_regular.prepare_hu_turn2_stage8b_counterfactual_loss_targets")
    foreach ($decisionLog in $decisionLogs) {
        $lossArgs += @("--decision-log", $decisionLog)
    }
    $lossArgs += @(
        "--local-replay-summary", (Join-Path $allFiredReplayOutputDir "topk_hard_negative_replay_summary.csv"),
        "--include-non-loss-controls",
        "--output-dir", $lossTargetsAllFiredOutputDir
    )
    python @lossArgs
    if ($LASTEXITCODE -ne 0) { throw "All-fired loss target extraction failed with exit code $LASTEXITCODE" }
    $lossTargetDirs += $lossTargetsAllFiredOutputDir
}

if (-not $SkipFalsePositiveReplay) {
    $replayArgs = @(
        "-m", "ofc_regular.replay_hu_turn2_stage8b_topk_hard_negatives",
        "--input-jsonl", (Join-Path $replayPackOutputDir "topk_false_positive_hard_negatives.jsonl"),
        "--future-samples", "$FalsePositiveFutureSamples",
        "--seed", "$($ReplaySeed + 1)",
        "--offset", "$FalsePositiveReplayOffset",
        "--prediction-threads", "$PredictionThreads",
        "--opening-lookahead-samples", "$OpeningLookaheadSamples",
        "--t3-continuation", "$T3Continuation",
        "--output-dir", "$falsePositiveReplayOutputDir"
    )
    if ($FalsePositiveReplayLimit -gt 0) { $replayArgs += @("--limit", "$FalsePositiveReplayLimit") }
    python @replayArgs
    if ($LASTEXITCODE -ne 0) { throw "False-positive replay failed with exit code $LASTEXITCODE" }

    $lossArgs = @("-m", "ofc_regular.prepare_hu_turn2_stage8b_counterfactual_loss_targets")
    foreach ($decisionLog in $decisionLogs) {
        $lossArgs += @("--decision-log", $decisionLog)
    }
    $lossArgs += @(
        "--local-replay-summary", (Join-Path $falsePositiveReplayOutputDir "topk_hard_negative_replay_summary.csv"),
        "--output-dir", $lossTargetsFalsePositiveOutputDir
    )
    python @lossArgs
    if ($LASTEXITCODE -ne 0) { throw "False-positive loss target extraction failed with exit code $LASTEXITCODE" }
    $lossTargetDirs += $lossTargetsFalsePositiveOutputDir
}

if (-not $SkipLossTargetCollection -and $lossTargetDirs.Count -gt 0) {
    $collectArgs = @("-m", "ofc_regular.collect_hu_turn2_stage8b_loss_targets")
    foreach ($lossTargetDir in $lossTargetDirs) {
        $collectArgs += @("--input-dir", $lossTargetDir)
    }
    $collectArgs += @("--output-dir", $lossTargetCollectionOutputDir)
    python @collectArgs
    if ($LASTEXITCODE -ne 0) { throw "Loss target collection failed with exit code $LASTEXITCODE" }

    if (-not $SkipRiskTargetAudit) {
        python -m ofc_regular.analyze_hu_turn2_stage8b_risk_targets `
            --collection-dir $lossTargetCollectionOutputDir `
            --output-dir $riskTargetAuditOutputDir
        if ($LASTEXITCODE -ne 0) { throw "Risk target audit failed with exit code $LASTEXITCODE" }
    }

    if (-not $SkipRiskTargetGap) {
        python -m ofc_regular.analyze_hu_turn2_stage8c_risk_target_gap `
            --collection-dir $lossTargetCollectionOutputDir `
            --output-dir $riskTargetGapOutputDir
        if ($LASTEXITCODE -ne 0) { throw "Risk target gap analysis failed with exit code $LASTEXITCODE" }
    }
}

[pscustomobject]@{
    input_dirs = $resolvedInputDirs
    decision_logs = $decisionLogs
    output_root = $OutputRoot
    aggregate_output_dir = $aggregateOutputDir
    replay_pack_output_dir = $replayPackOutputDir
    all_fired_replay_output_dir = if ($SkipAllFiredReplay) { "" } else { $allFiredReplayOutputDir }
    false_positive_replay_output_dir = if ($SkipFalsePositiveReplay) { "" } else { $falsePositiveReplayOutputDir }
    all_fired_replay_offset = $AllFiredReplayOffset
    all_fired_replay_limit = $AllFiredReplayLimit
    false_positive_replay_offset = $FalsePositiveReplayOffset
    false_positive_replay_limit = $FalsePositiveReplayLimit
    loss_target_collection_output_dir = if ($SkipLossTargetCollection) { "" } else { $lossTargetCollectionOutputDir }
    risk_target_audit_output_dir = if ($SkipLossTargetCollection -or $SkipRiskTargetAudit) { "" } else { $riskTargetAuditOutputDir }
    risk_target_gap_output_dir = if ($SkipLossTargetCollection -or $SkipRiskTargetGap) { "" } else { $riskTargetGapOutputDir }
    production_p2_fixed = "No-Go"
    t1_training = "No-Go"
    teacher_50k = "No-Go"
} | ConvertTo-Json -Depth 6
