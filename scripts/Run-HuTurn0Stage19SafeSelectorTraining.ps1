param(
    [string]$Mc32Input = "outputs/hu_turn0_stage19_safe_selector_replay/mc32_aggregate/hu_turn0_fired_pair_replay.jsonl",
    [Parameter(Mandatory = $true)]
    [string]$Mc512Input,
    [string]$OutputDir = "outputs/training/hu_turn0_stage19_safe_selector",
    [string]$ModelDir = "models",
    [double]$BroadHighMcWeight = 4.0,
    [int]$Folds = 5
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    if (-not (Test-Path -LiteralPath $Mc32Input)) { throw "MC32 input not found: $Mc32Input" }
    if (-not (Test-Path -LiteralPath $Mc512Input)) { throw "MC512 input not found: $Mc512Input" }
    if ($BroadHighMcWeight -le 0.0) { throw "BroadHighMcWeight must be positive" }
    if ($Folds -lt 2) { throw "Folds must be at least two" }

    $datasetDir = Join-Path $OutputDir "dataset"
    & python -m ofc_regular.prepare_hu_turn0_safe_selector_dataset `
        --mc32 $Mc32Input `
        --mc512 $Mc512Input `
        --output-dir $datasetDir
    if ($LASTEXITCODE -ne 0) { throw "T0 selector dataset preparation failed" }

    $datasets = @(
        [pscustomobject]@{
            Name = "highmc"
            Path = (Join-Path $datasetDir "selector_high_mc_only.jsonl")
            HighMcWeight = 1.0
        },
        [pscustomobject]@{
            Name = "broad_refined"
            Path = (Join-Path $datasetDir "selector_broad_with_mc512_overrides.jsonl")
            HighMcWeight = $BroadHighMcWeight
        }
    )
    $featureModes = @(
        "meta_only",
        "compact_delta_plus_meta",
        "compact_candidate_delta_plus_meta",
        "delta_plus_meta",
        "candidate_delta_plus_meta"
    )
    $runs = @()
    foreach ($dataset in $datasets) {
        foreach ($mode in $featureModes) {
            $runName = ("{0}_{1}" -f $dataset.Name, $mode)
            $runOutput = Join-Path $OutputDir $runName
            $modelOutput = Join-Path $ModelDir ("hu_turn0_stage19_safe_selector_{0}.pkl" -f $runName)
            & python -m ofc_regular.train_hu_turn0_safe_override_selector `
                --input $dataset.Path `
                --output-dir $runOutput `
                --model-output $modelOutput `
                --feature-mode $mode `
                --allowed-seats first `
                --high-mc-weight $dataset.HighMcWeight `
                --folds $Folds
            if ($LASTEXITCODE -ne 0) { throw "T0 selector training failed: $runName" }
            $summary = Get-Content (Join-Path $runOutput "summary.json") -Raw | ConvertFrom-Json
            $runs += [pscustomobject]@{
                run = $runName
                dataset = $dataset.Name
                feature_mode = $mode
                rows = $summary.rows
                positive = $summary.positive
                negative = $summary.negative
                gray = $summary.gray
                best_model = $summary.best_model
                best_oof_average_precision = $summary.best_oof_average_precision
                model = $modelOutput
                threshold_sweep = (Join-Path $runOutput "threshold_sweep_oof.csv")
            }
        }
    }
    $summaryPath = Join-Path $OutputDir "training_candidates.json"
    $runs | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $summaryPath -Encoding utf8
    $selectionDir = Join-Path $OutputDir "selection"
    & python -m ofc_regular.select_hu_turn0_safe_selector_candidate `
        --training-candidates $summaryPath `
        --output-dir $selectionDir
    if ($LASTEXITCODE -ne 0) { throw "T0 selector candidate selection failed" }
    $selection = Get-Content (Join-Path $selectionDir "selection.json") -Raw | ConvertFrom-Json
    [pscustomobject]@{
        schema = "hu_turn0_stage19_safe_selector_training_run_v1"
        mc32_input = $Mc32Input
        mc512_input = $Mc512Input
        dataset_summary = (Join-Path $datasetDir "summary.json")
        candidates = $runs
        selection = $selection
        runtime_status = "not_approved_until_fixed_fresh_whole_game_holdout"
    } | ConvertTo-Json -Depth 8
}
finally {
    Pop-Location
}
