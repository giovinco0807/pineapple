param(
    [string]$CacheDir = "outputs\hu_turn2_current_fl_ev_stage8c_mixed5k_mc16_plus_8seed_all_fired_replay_mc128_cache",
    [string]$SeedModel = "models\hu_turn2_stage8b_safe_lcb196_20k_reference_override_cached_rank_wide.pt",
    [string]$InitFromModel = "",
    [ValidateSet("init_model", "cache")]
    [string]$InitStatsSource = "init_model",
    [string]$OutputRoot = "outputs\training\hu_turn2_stage9_candidate_generator_current_fl_5k",
    [string]$ModelOutput = "models\hu_turn2_stage9_candidate_generator_current_fl_5k.pt",
    [int]$Epochs = 20,
    [double]$LearningRate = 0.0005,
    [ValidateSet("val_avg_regret", "val_top3_recall", "val_top5_avg_regret", "val_top5_recall", "val_top10_avg_regret", "val_top10_recall")]
    [string]$EarlyStopMetric = "val_top5_avg_regret",
    [string]$HiddenLayerSizes = "1024,512,256",
    [string]$Device = "auto",
    [int]$BatchActionRows = 32768,
    [int]$HardMissTopK = 5,
    [int]$HardNegativeTopK = 5,
    [double]$MissedPositiveWeight = 4.0,
    [double]$HardNegativeWeight = 3.0,
    [double]$OracleBestWeight = 1.0,
    [double]$RankingLossWeight = 0.05,
    [double]$ListwiseLossWeight = 0.05
)

$ErrorActionPreference = "Stop"

$preAudit = Join-Path $OutputRoot "pretrain_audit"
$trainingDir = Join-Path $OutputRoot "training"
$postAudit = Join-Path $OutputRoot "posttrain_audit"

python -m ofc_regular.prepare_hu_turn2_stage9_candidate_generator_cache `
    --cache-dir $CacheDir `
    --model $SeedModel `
    --output-dir $preAudit `
    --score-head ev `
    --topk 1,3,5,10 `
    --hard-miss-topk $HardMissTopK `
    --hard-negative-topk $HardNegativeTopK `
    --missed-positive-weight $MissedPositiveWeight `
    --hard-negative-weight $HardNegativeWeight `
    --oracle-best-weight $OracleBestWeight `
    --device cpu

$candidateRows = Join-Path $preAudit "candidate_generator_training_rows.jsonl"

$trainArgs = @(
    "-m", "ofc_regular.train_hu_turn2_pilot_model",
    "--cache-dir", $CacheDir,
    "--candidate-generator-training-jsonl", $candidateRows,
    "--model-output", $ModelOutput,
    "--output-dir", $trainingDir,
    "--epochs", $Epochs,
    "--learning-rate", $LearningRate,
    "--early-stop-metric", $EarlyStopMetric,
    "--batch-action-rows", $BatchActionRows,
    "--hidden-layer-sizes", $HiddenLayerSizes,
    "--ranking-loss-weight", $RankingLossWeight,
    "--listwise-loss-weight", $ListwiseLossWeight,
    "--dropout", 0.05,
    "--device", $Device,
    "--log-every", 1
)
if ($InitFromModel -ne "") {
    $trainArgs += @("--init-from-model", $InitFromModel, "--init-stats-source", $InitStatsSource)
}
python @trainArgs

python -m ofc_regular.prepare_hu_turn2_stage9_candidate_generator_cache `
    --cache-dir $CacheDir `
    --model $ModelOutput `
    --output-dir $postAudit `
    --score-head ev `
    --topk 1,3,5,10 `
    --hard-miss-topk $HardMissTopK `
    --hard-negative-topk $HardNegativeTopK `
    --missed-positive-weight $MissedPositiveWeight `
    --hard-negative-weight $HardNegativeWeight `
    --oracle-best-weight $OracleBestWeight `
    --device cpu

Write-Host "Pretrain audit: $preAudit"
Write-Host "Training dir:    $trainingDir"
Write-Host "Posttrain audit: $postAudit"
Write-Host "Model:           $ModelOutput"
