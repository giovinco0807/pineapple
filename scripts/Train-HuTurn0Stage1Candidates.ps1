param(
    [Parameter(Mandatory = $true)]
    [Alias("Input")]
    [string]$TeacherInput,
    [string]$ValidationInput = "",
    [string]$BroadValidationInput = "",
    [string]$OutputDir = "outputs/training/hu_turn0_stage1_all232_mc4_100",
    [string]$ModelDir = "models",
    [string]$ArtifactStem = "hu_turn0_stage1_all232_mc4_100",
    [string]$OpeningModel = "models/opening_stage7_torch_wide.pt",
    [int]$Seed = 2026103101,
    [double]$Holdout = 0.2,
    [int]$SuitAugmentations = 3,
    [switch]$SkipExtraTrees,
    [ValidateSet("negative_regret", "baseline_delta", "absolute_score")]
    [string]$RegressionTarget = "negative_regret",
    [string[]]$SourceWeights = @(
        "hu_turn0_terminal_rollout_mc1=1.0",
        "hu_turn0_terminal_rollout_mc4=4.0"
    )
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
    New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null

    $holdoutPath = Join-Path $OutputDir "holdout.jsonl"
    $targetLabel = if ($RegressionTarget -eq "negative_regret") {
        "regret"
    }
    else {
        $RegressionTarget -replace '_', '-'
    }
    $hgbModel = Join-Path $ModelDir "${ArtifactStem}_hgb_${targetLabel}_aug${SuitAugmentations}.pkl"
    $extraTreesModel = Join-Path $ModelDir "${ArtifactStem}_extra_trees_${targetLabel}_aug${SuitAugmentations}.pkl"
    $classifierModel = Join-Path $ModelDir "${ArtifactStem}_hgb_nearbest_aug${SuitAugmentations}.pkl"
    $listwiseModel = Join-Path $ModelDir "${ArtifactStem}_listwise_torch_aug${SuitAugmentations}.pt"

    function Invoke-Trainer {
        param([string[]]$Arguments)
        & python -m ofc_regular.train_hu_turn1_candidate_generator @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "T0 candidate training failed with exit code $LASTEXITCODE"
        }
    }

    $common = @(
        "--input", $TeacherInput,
        "--seed", "$Seed",
        "--suit-augmentations", "$SuitAugmentations",
        "--accept-regret", "0.25",
        "--gray-regret", "2.0"
    )
    if ($ValidationInput) {
        $common += @("--validation-input", $ValidationInput)
        $evaluationHoldout = $ValidationInput
    }
    else {
        $common += @("--holdout", "$Holdout", "--holdout-output", $holdoutPath)
        $evaluationHoldout = $holdoutPath
    }
    foreach ($sourceWeight in $SourceWeights) {
        if ($sourceWeight) {
            $common += @("--source-weight", $sourceWeight)
        }
    }

    Invoke-Trainer ($common + @(
        "--model-output", $hgbModel,
        "--metrics-output", (Join-Path $OutputDir "hgb_regret_metrics.json"),
        "--model-type", "hgb_regressor",
        "--regression-target", $RegressionTarget,
        "--max-iter", "500",
        "--max-leaf-nodes", "63",
        "--l2", "2.0"
    ))

    $trainedModels = @($hgbModel)
    if (-not $SkipExtraTrees) {
        Invoke-Trainer ($common + @(
            "--model-output", $extraTreesModel,
            "--metrics-output", (Join-Path $OutputDir "extra_trees_regret_metrics.json"),
            "--model-type", "extra_trees_regressor",
            "--regression-target", $RegressionTarget,
            "--n-estimators", "400",
            "--max-depth", "24",
            "--min-samples-leaf", "8"
        ))
        $trainedModels += $extraTreesModel
    }

    Invoke-Trainer ($common + @(
        "--model-output", $classifierModel,
        "--metrics-output", (Join-Path $OutputDir "hgb_nearbest_metrics.json"),
        "--model-type", "hgb",
        "--max-iter", "400",
        "--max-leaf-nodes", "63",
        "--l2", "2.0",
        "--positive-weight", "6.0"
    ))

    $listwiseArgs = @(
        "--input", $TeacherInput,
        "--model-output", $listwiseModel,
        "--metrics-output", (Join-Path $OutputDir "listwise_torch_metrics.json"),
        "--seed", "$Seed",
        "--suit-augmentations", "$SuitAugmentations",
        "--epochs", "50",
        "--batch-size", "32",
        "--hidden-sizes", "512,256,128",
        "--dropout", "0.05",
        "--learning-rate", "0.001",
        "--weight-decay", "0.0001",
        "--listwise-temperature", "4.0",
        "--mse-weight", "0.1",
        "--early-stopping-patience", "6"
    )
    if ($ValidationInput) {
        $listwiseArgs += @("--validation-input", $ValidationInput)
    }
    else {
        $listwiseArgs += @("--holdout", "$Holdout")
    }
    foreach ($sourceWeight in $SourceWeights) {
        if ($sourceWeight) {
            $listwiseArgs += @("--source-weight", $sourceWeight)
        }
    }
    & python -m ofc_regular.train_hu_turn1_listwise_torch @listwiseArgs
    if ($LASTEXITCODE -ne 0) {
        throw "T0 listwise candidate training failed with exit code $LASTEXITCODE"
    }
    $trainedModels += @($classifierModel, $listwiseModel)

    $coverageDir = Join-Path $OutputDir "coverage_union"
    & python -m ofc_regular.analyze_hu_turn1_candidate_coverage `
        --input $evaluationHoldout `
        --candidate-models $trainedModels `
        --candidate-topk 20 `
        --union-mode rank_sum `
        --ks 5,10,15,20,30,40,60 `
        --output-dir $coverageDir
    if ($LASTEXITCODE -ne 0) {
        throw "T0 candidate coverage analysis failed with exit code $LASTEXITCODE"
    }

    $coverageMinRankDir = Join-Path $OutputDir "coverage_union_min_rank"
    & python -m ofc_regular.analyze_hu_turn1_candidate_coverage `
        --input $evaluationHoldout `
        --candidate-models $trainedModels `
        --candidate-topk 20 `
        --union-mode min_rank `
        --ks 5,10,15,20,30,40,60,80 `
        --output-dir $coverageMinRankDir
    if ($LASTEXITCODE -ne 0) {
        throw "T0 min-rank candidate coverage analysis failed with exit code $LASTEXITCODE"
    }

    $openingCoverageDir = Join-Path $OutputDir "coverage_old_opening"
    & python -m ofc_regular.analyze_hu_turn1_candidate_coverage `
        --input $evaluationHoldout `
        --candidate-model $OpeningModel `
        --ks 1,3,5,10,20,30,40,60,100 `
        --output-dir $openingCoverageDir
    if ($LASTEXITCODE -ne 0) {
        throw "T0 opening-model coverage analysis failed with exit code $LASTEXITCODE"
    }

    $combinedCoverageDir = Join-Path $OutputDir "coverage_old_plus_hu_union"
    $combinedModels = @($OpeningModel) + $trainedModels
    & python -m ofc_regular.analyze_hu_turn1_candidate_coverage `
        --input $evaluationHoldout `
        --candidate-models $combinedModels `
        --candidate-topk 20 `
        --union-mode rank_sum `
        --ks 10,20,30,40,50,60,80 `
        --output-dir $combinedCoverageDir
    if ($LASTEXITCODE -ne 0) {
        throw "T0 combined candidate coverage analysis failed with exit code $LASTEXITCODE"
    }

    $combinedMinRankCoverageDir = Join-Path $OutputDir "coverage_old_plus_hu_min_rank"
    & python -m ofc_regular.analyze_hu_turn1_candidate_coverage `
        --input $evaluationHoldout `
        --candidate-models $combinedModels `
        --candidate-topk 20 `
        --union-mode min_rank `
        --ks 10,20,30,40,50,60,80,100 `
        --output-dir $combinedMinRankCoverageDir
    if ($LASTEXITCODE -ne 0) {
        throw "T0 combined min-rank coverage analysis failed with exit code $LASTEXITCODE"
    }

    $broadCoverage = $null
    if ($BroadValidationInput) {
        $broadCoverageDir = Join-Path $OutputDir "coverage_broad_mc1"
        & python -m ofc_regular.analyze_hu_turn1_candidate_coverage `
            --input $BroadValidationInput `
            --candidate-models $combinedModels `
            --candidate-topk 20 `
            --union-mode rank_sum `
            --ks 10,20,30,40,50,60,80,100 `
            --output-dir $broadCoverageDir
        if ($LASTEXITCODE -ne 0) {
            throw "T0 broad MC1 coverage analysis failed with exit code $LASTEXITCODE"
        }
        $broadCoverage = Join-Path $broadCoverageDir "coverage_summary.json"
    }

    [pscustomobject]@{
        input = $TeacherInput
        holdout = $evaluationHoldout
        broad_holdout = $BroadValidationInput
        source_weights = $SourceWeights
        regression_target = $RegressionTarget
        hgb_regret_model = $hgbModel
        extra_trees_regret_model = if ($SkipExtraTrees) { $null } else { $extraTreesModel }
        extra_trees_skipped = [bool]$SkipExtraTrees
        hgb_nearbest_model = $classifierModel
        listwise_torch_model = $listwiseModel
        coverage = Join-Path $coverageDir "coverage_summary.json"
        coverage_min_rank = Join-Path $coverageMinRankDir "coverage_summary.json"
        opening_coverage = Join-Path $openingCoverageDir "coverage_summary.json"
        combined_coverage = Join-Path $combinedCoverageDir "coverage_summary.json"
        combined_min_rank_coverage = Join-Path $combinedMinRankCoverageDir "coverage_summary.json"
        broad_coverage = $broadCoverage
        decision = "candidate_generator_only_not_p0"
    } | ConvertTo-Json -Depth 4
}
finally {
    Pop-Location
}
