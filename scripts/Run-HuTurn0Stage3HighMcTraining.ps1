param(
    [Parameter(Mandatory = $true)]
    [string[]]$TeacherInputs,
    [string]$DatasetDir = "outputs/hu_turn0_stage3_high_mc_dataset",
    [string]$TrainingDir = "outputs/training/hu_turn0_stage3_high_mc",
    [string]$CalibrationDir = "outputs/evals/hu_turn0_stage3_high_mc_calibration",
    [string]$ModelDir = "models",
    [string]$ArtifactStem = "hu_turn0_stage3_top60_mc32",
    [int]$Seed = 2026105101,
    [double]$Holdout = 0.20,
    [int]$SuitAugmentations = 3,
    [double]$DeltaSeWeightFloor = 1.0,
    [switch]$SkipExtraTrees
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    & python -m ofc_regular.prepare_hu_turn0_high_mc_dataset `
        --input $TeacherInputs `
        --output-dir $DatasetDir `
        --holdout $Holdout `
        --seed $Seed `
        --min-future-samples 32 `
        --expected-evaluated-actions 60
    if ($LASTEXITCODE -ne 0) {
        throw "T0 high-MC dataset preparation failed with exit code $LASTEXITCODE"
    }

    $trainInput = Join-Path $DatasetDir "train.jsonl"
    $validationInput = Join-Path $DatasetDir "holdout.jsonl"
    $sourceWeights = @(
        "hu_turn0_terminal_rollout_mc32=1.0",
        "hu_turn0_terminal_rollout_mc64=2.0",
        "hu_turn0_terminal_rollout_mc128=4.0"
    )
    & (Join-Path $PSScriptRoot "Train-HuTurn0Stage1Candidates.ps1") `
        -TeacherInput $trainInput `
        -ValidationInput $validationInput `
        -OutputDir $TrainingDir `
        -ModelDir $ModelDir `
        -ArtifactStem $ArtifactStem `
        -Seed $Seed `
        -SuitAugmentations $SuitAugmentations `
        -SkipExtraTrees:$SkipExtraTrees `
        -RegressionTarget baseline_delta `
        -SourceWeights $sourceWeights
    if ($LASTEXITCODE -ne 0) {
        throw "T0 high-MC model training failed with exit code $LASTEXITCODE"
    }

    $hgbModel = Join-Path $ModelDir "${ArtifactStem}_hgb_baseline-delta_aug${SuitAugmentations}.pkl"
    $extraTreesModel = Join-Path $ModelDir "${ArtifactStem}_extra_trees_baseline-delta_aug${SuitAugmentations}.pkl"
    $classifierModel = Join-Path $ModelDir "${ArtifactStem}_hgb_nearbest_aug${SuitAugmentations}.pkl"
    $listwiseModel = Join-Path $ModelDir "${ArtifactStem}_listwise_torch_aug${SuitAugmentations}.pt"
    $candidateModels = @($hgbModel, $classifierModel, $listwiseModel)
    if (-not $SkipExtraTrees) {
        $candidateModels += $extraTreesModel
    }
    $unaugmentedHgbModel = $null
    if ($SuitAugmentations -gt 0) {
        $unaugmentedHgbModel = Join-Path $ModelDir "${ArtifactStem}_hgb_baseline-delta_aug0.pkl"
        $unaugmentedArgs = @(
            "-m", "ofc_regular.train_hu_turn1_candidate_generator",
            "--input", $trainInput,
            "--validation-input", $validationInput,
            "--model-output", $unaugmentedHgbModel,
            "--metrics-output", (Join-Path $TrainingDir "hgb_baseline_delta_unaugmented_metrics.json"),
            "--model-type", "hgb_regressor",
            "--regression-target", "baseline_delta",
            "--seed", ([string]$Seed),
            "--suit-augmentations", "0",
            "--accept-regret", "0.25",
            "--gray-regret", "2.0",
            "--max-iter", "500",
            "--max-leaf-nodes", "63",
            "--l2", "2.0"
        )
        foreach ($sourceWeight in $sourceWeights) {
            $unaugmentedArgs += @("--source-weight", $sourceWeight)
        }
        & python @unaugmentedArgs
        if ($LASTEXITCODE -ne 0) {
            throw "T0 unaugmented HGB comparison training failed with exit code $LASTEXITCODE"
        }
        $candidateModels += $unaugmentedHgbModel
    }
    $weightedHgbModel = $null
    if ($DeltaSeWeightFloor -gt 0.0) {
        $floorTag = $DeltaSeWeightFloor.ToString("0.###", [Globalization.CultureInfo]::InvariantCulture).Replace(".", "p")
        $weightedHgbModel = Join-Path $ModelDir "${ArtifactStem}_hgb_baseline-delta_sew${floorTag}_aug${SuitAugmentations}.pkl"
        $weightedArgs = @(
            "-m", "ofc_regular.train_hu_turn1_candidate_generator",
            "--input", $trainInput,
            "--validation-input", $validationInput,
            "--model-output", $weightedHgbModel,
            "--metrics-output", (Join-Path $TrainingDir "hgb_baseline_delta_se_weighted_metrics.json"),
            "--model-type", "hgb_regressor",
            "--regression-target", "baseline_delta",
            "--delta-se-weight-floor", ([string]$DeltaSeWeightFloor),
            "--seed", ([string]$Seed),
            "--suit-augmentations", ([string]$SuitAugmentations),
            "--accept-regret", "0.25",
            "--gray-regret", "2.0",
            "--max-iter", "500",
            "--max-leaf-nodes", "63",
            "--l2", "2.0"
        )
        foreach ($sourceWeight in $sourceWeights) {
            $weightedArgs += @("--source-weight", $sourceWeight)
        }
        & python @weightedArgs
        if ($LASTEXITCODE -ne 0) {
            throw "T0 uncertainty-weighted HGB training failed with exit code $LASTEXITCODE"
        }
        $candidateModels += $weightedHgbModel
    }

    $safeLcbModel = Join-Path $ModelDir "${ArtifactStem}_hgb_safe_lcb196_aug${SuitAugmentations}.pkl"
    $safeLcbArgs = @(
        "-m", "ofc_regular.train_hu_turn1_candidate_generator",
        "--input", $trainInput,
        "--validation-input", $validationInput,
        "--model-output", $safeLcbModel,
        "--metrics-output", (Join-Path $TrainingDir "hgb_safe_lcb196_metrics.json"),
        "--model-type", "hgb",
        "--classification-target", "safe_lcb196",
        "--seed", ([string]$Seed),
        "--suit-augmentations", ([string]$SuitAugmentations),
        "--max-iter", "500",
        "--max-leaf-nodes", "63",
        "--l2", "2.0",
        "--positive-weight", "12.0",
        "--gray-weight", "0.1",
        "--negative-weight", "1.0",
        "--hard-negative-weight", "3.0"
    )
    foreach ($sourceWeight in $sourceWeights) {
        $safeLcbArgs += @("--source-weight", $sourceWeight)
    }
    & python @safeLcbArgs
    if ($LASTEXITCODE -ne 0) {
        throw "T0 safe-LCB classifier training failed with exit code $LASTEXITCODE"
    }
    $candidateModels += $safeLcbModel

    $calibrationArgs = @(
        "-m", "ofc_regular.analyze_hu_turn0_high_mc_gate",
        "--input", $validationInput,
        "--candidate-models"
    ) + $candidateModels + @(
        "--thresholds", "0,0.1,0.25,0.5,0.75,1,1.5,2,2.5,3",
        "--output-dir", $CalibrationDir
    )
    & python @calibrationArgs
    if ($LASTEXITCODE -ne 0) {
        throw "T0 high-MC gate calibration failed with exit code $LASTEXITCODE"
    }

    [pscustomobject]@{
        schema = "hu_turn0_stage3_high_mc_training_run_v1"
        teacher_inputs = $TeacherInputs
        dataset_summary = (Join-Path $DatasetDir "summary.json")
        training_dir = $TrainingDir
        calibration_summary = (Join-Path $CalibrationDir "summary.json")
        models = $candidateModels
        unaugmented_hgb_model = $unaugmentedHgbModel
        extra_trees_skipped = [bool]$SkipExtraTrees
        uncertainty_weight_floor = $DeltaSeWeightFloor
        safe_lcb_model = $safeLcbModel
        decision = "calibration_only_fresh_seat_swap_required"
    } | ConvertTo-Json -Depth 6
}
finally {
    Pop-Location
}
