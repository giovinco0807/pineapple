param(
    [Parameter(Mandatory = $true)]
    [string]$Mc1Input,
    [Parameter(Mandatory = $true)]
    [string]$Mc4Input,
    [string]$DatasetDir = "outputs/hu_turn0_stage2_candidate_dataset",
    [string]$TrainingDir = "outputs/training/hu_turn0_stage2_mixed500mc1_100mc4",
    [string]$ModelDir = "models",
    [string]$ArtifactStem = "hu_turn0_stage2_mixed500mc1_100mc4",
    [int]$Seed = 2026103101,
    [double]$Mc1Holdout = 0.10,
    [double]$Mc4Holdout = 0.20,
    [int]$SuitAugmentations = 1,
    [double]$Mc1Weight = 1.0,
    [double]$Mc4Weight = 4.0
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    & python -m ofc_regular.prepare_hu_turn0_candidate_dataset `
        --mc1-input $Mc1Input `
        --mc4-input $Mc4Input `
        --output-dir $DatasetDir `
        --mc1-holdout $Mc1Holdout `
        --mc4-holdout $Mc4Holdout `
        --seed $Seed
    if ($LASTEXITCODE -ne 0) {
        throw "T0 candidate dataset preparation failed with exit code $LASTEXITCODE"
    }

    $trainInput = Join-Path $DatasetDir "train_mixed_mc1_mc4.jsonl"
    $mc1Validation = Join-Path $DatasetDir "holdout_mc1.jsonl"
    $mc4Validation = Join-Path $DatasetDir "holdout_mc4.jsonl"
    $sourceWeights = @(
        ("hu_turn0_terminal_rollout_mc1={0}" -f $Mc1Weight),
        ("hu_turn0_terminal_rollout_mc4={0}" -f $Mc4Weight)
    )

    & (Join-Path $PSScriptRoot "Train-HuTurn0Stage1Candidates.ps1") `
        -TeacherInput $trainInput `
        -ValidationInput $mc4Validation `
        -BroadValidationInput $mc1Validation `
        -OutputDir $TrainingDir `
        -ModelDir $ModelDir `
        -ArtifactStem $ArtifactStem `
        -Seed $Seed `
        -SuitAugmentations $SuitAugmentations `
        -SourceWeights $sourceWeights
    if ($LASTEXITCODE -ne 0) {
        throw "T0 Stage2 candidate training failed with exit code $LASTEXITCODE"
    }

    [pscustomobject]@{
        schema = "hu_turn0_stage2_candidate_training_run_v1"
        mc1_input = $Mc1Input
        mc4_input = $Mc4Input
        dataset_summary = (Join-Path $DatasetDir "summary.json")
        training_dir = $TrainingDir
        artifact_stem = $ArtifactStem
        primary_validation = $mc4Validation
        broad_validation = $mc1Validation
        source_weights = $sourceWeights
        decision = "candidate_generator_comparison_only_not_p0"
    } | ConvertTo-Json -Depth 5
}
finally {
    Pop-Location
}
