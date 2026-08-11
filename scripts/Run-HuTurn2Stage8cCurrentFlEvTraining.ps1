param(
    [string]$CacheDir = "outputs/hu_turn2_current_fl_ev_stage8c_mixed5k_mc16_cache",
    [string]$OutputDir = "outputs/training/hu_turn2_current_fl_ev_stage8c_mixed5k_mc16",
    [string]$ModelOutput = "models/hu_turn2_current_fl_ev_stage8c_mixed5k_mc16.pt",
    [int]$Epochs = 30,
    [int]$BatchActionRows = 32768,
    [double]$LearningRate = 0.0005,
    [double]$WeightDecay = 0.00005,
    [string]$HiddenLayerSizes = "1024,512,256",
    [double]$Dropout = 0.05,
    [double]$RankingLossWeight = 0.05,
    [double]$ListwiseLossWeight = 0.05,
    [double]$GateLossWeight = 1.0,
    [double]$GateNegativeWeight = 2.0,
    [string[]]$AuxiliarySourceBucket = @(),
    [double]$AuxiliaryRegressionWeight = 0.0,
    [double]$AuxiliaryRankingWeight = 0.0,
    [double]$AuxiliaryListwiseWeight = 0.0,
    [double]$AuxiliaryGateWeight = 1.0,
    [int]$Patience = 5,
    [int]$Seed = 2026063002,
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"

$arguments = @(
    "-m", "ofc_regular.train_hu_turn2_pilot_model",
    "--cache-dir", $CacheDir,
    "--output-dir", $OutputDir,
    "--model-output", $ModelOutput,
    "--epochs", "$Epochs",
    "--batch-action-rows", "$BatchActionRows",
    "--learning-rate", "$LearningRate",
    "--weight-decay", "$WeightDecay",
    "--hidden-layer-sizes", $HiddenLayerSizes,
    "--dropout", "$Dropout",
    "--ranking-loss-weight", "$RankingLossWeight",
    "--listwise-loss-weight", "$ListwiseLossWeight",
    "--gate-loss-weight", "$GateLossWeight",
    "--gate-negative-weight", "$GateNegativeWeight",
    "--patience", "$Patience",
    "--seed", "$Seed",
    "--device", $Device
)

foreach ($bucket in $AuxiliarySourceBucket) {
    if ($bucket) {
        $arguments += @("--auxiliary-source-bucket", $bucket)
    }
}
if ($AuxiliarySourceBucket.Count -gt 0) {
    $arguments += @(
        "--auxiliary-regression-weight", "$AuxiliaryRegressionWeight",
        "--auxiliary-ranking-weight", "$AuxiliaryRankingWeight",
        "--auxiliary-listwise-weight", "$AuxiliaryListwiseWeight",
        "--auxiliary-gate-weight", "$AuxiliaryGateWeight"
    )
}

python @arguments
