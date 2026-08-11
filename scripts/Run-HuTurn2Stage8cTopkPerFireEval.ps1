param(
    [int]$GamesPerSeed = 2000,
    [int]$TargetRealizedOverridesPerSeed = 50,
    [int]$TargetRiskVetoesPerSeed = 0,
    [int]$TargetFireSelectorRejectionsPerSeed = 0,
    [string]$Seeds = "2026063101,2026063102,2026063103",
    [int]$SeedStride = 1000000,
    [string]$Configs = "k3/mc64/d0/se0/confirm128/cse1/pd0/seat=first/bygate_delta,k5/mc64/d0/se0/confirm128/cse1/pd0/seat=first/bygate_delta",
    [string]$Model = "models\hu_turn2_current_fl_ev_stage8c_mixed5k_mc16.pt",
    [string]$OutputDir = "outputs\evals\hu_turn2_current_fl_ev_stage8c_topk_confirm128_cse1_fire_target",
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Device = "auto",
    [int]$PredictionThreads = 1,
    [int]$OpeningLookaheadSamples = 64,
    [int]$ProgressEvery = 25,
    [ValidateSet("stage3_reference_default", "stage7_m5_r10")]
    [string]$T3Continuation = "stage3_reference_default",
    [string]$Stage8cRiskModel = "",
    [double]$Stage8cRiskThreshold = 0.35,
    [int]$Stage8cRiskRankMin = 0,
    [int]$Stage8cRiskRankMax = 0,
    [switch]$Stage8cRiskAuditOnly,
    [string]$Stage8cFireSelectorModel = "",
    [double]$Stage8cFireSelectorThreshold = 0.7,
    [switch]$Stage8cFireSelectorAuditOnly,
    [switch]$Stage8cFireSelectorDirectFire,
    [switch]$NoDecisionLog
)

$ErrorActionPreference = "Stop"

if ($Stage8cFireSelectorDirectFire -and $Stage8cFireSelectorModel -eq "") {
    throw "Stage8cFireSelectorDirectFire requires Stage8cFireSelectorModel"
}
if ($Stage8cFireSelectorDirectFire -and $Stage8cFireSelectorAuditOnly) {
    throw "Stage8cFireSelectorDirectFire cannot be combined with Stage8cFireSelectorAuditOnly"
}

$arguments = @(
    "-m", "ofc_regular.evaluate_hu_turn2_stage8b_topk_mc_rerank",
    "--games-per-seed", "$GamesPerSeed",
    "--target-realized-overrides-per-seed", "$TargetRealizedOverridesPerSeed",
    "--target-risk-vetoes-per-seed", "$TargetRiskVetoesPerSeed",
    "--target-fire-selector-rejections-per-seed", "$TargetFireSelectorRejectionsPerSeed",
    "--seeds", $Seeds,
    "--seed-stride", "$SeedStride",
    "--configs", $Configs,
    "--hu-turn2-stage8b-model", $Model,
    "--output-dir", $OutputDir,
    "--device", $Device,
    "--prediction-threads", "$PredictionThreads",
    "--opening-lookahead-samples", "$OpeningLookaheadSamples",
    "--t3-continuation", "$T3Continuation",
    "--progress-every", "$ProgressEvery"
)

if (-not $NoDecisionLog) {
    $arguments += "--write-decision-log"
}
if ($Stage8cRiskModel -ne "") {
    $arguments += @(
        "--stage8c-risk-model", $Stage8cRiskModel,
        "--stage8c-risk-threshold", "$Stage8cRiskThreshold"
    )
    if ($Stage8cRiskRankMin -gt 0) {
        $arguments += @("--stage8c-risk-rank-min", "$Stage8cRiskRankMin")
    }
    if ($Stage8cRiskRankMax -gt 0) {
        $arguments += @("--stage8c-risk-rank-max", "$Stage8cRiskRankMax")
    }
    if ($Stage8cRiskAuditOnly) {
        $arguments += "--stage8c-risk-audit-only"
    }
}
if ($Stage8cFireSelectorModel -ne "") {
    $arguments += @(
        "--stage8c-fire-selector-model", $Stage8cFireSelectorModel,
        "--stage8c-fire-selector-threshold", "$Stage8cFireSelectorThreshold"
    )
    if ($Stage8cFireSelectorAuditOnly) {
        $arguments += "--stage8c-fire-selector-audit-only"
    }
    if ($Stage8cFireSelectorDirectFire) {
        $arguments += "--stage8c-fire-selector-direct-fire"
    }
}

python @arguments
