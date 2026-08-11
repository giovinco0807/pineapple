param(
    [string]$ShardDir = "outputs/hu_turn2_current_fl_ev_stage8c_mixed5k_mc16/shards",
    [string[]]$ExtraInput = @(
        "outputs/evals/hu_turn2_stage8b_topk_confirm128_cse1_pd0_score_hard_negative_replay/topk_hard_negative_replay_teacher.jsonl"
    ),
    [string]$CacheDir = "outputs/hu_turn2_current_fl_ev_stage8c_mixed5k_mc16_cache",
    [string]$DType = "float32",
    [int]$Seed = 2026063001,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $ShardDir)) {
    throw "ShardDir not found: $ShardDir"
}

$inputFiles = @(
    Get-ChildItem -LiteralPath $ShardDir -Filter "*.jsonl" -File |
        Where-Object { $_.Name -notlike "*_final_turn_slow_states_top50.jsonl" } |
        Sort-Object FullName |
        ForEach-Object { $_.FullName }
)

foreach ($extra in $ExtraInput) {
    if ($extra -and (Test-Path -LiteralPath $extra)) {
        $inputFiles += (Resolve-Path -LiteralPath $extra).Path
    }
    elseif ($extra) {
        Write-Warning "ExtraInput not found, skipping: $extra"
    }
}

if ($inputFiles.Count -le 0) {
    throw "No JSONL input files found under $ShardDir"
}

$argsList = @("-m", "ofc_regular.build_hu_turn2_pilot_feature_cache")
foreach ($inputFile in $inputFiles) {
    $argsList += @("--input", $inputFile)
}
$argsList += @(
    "--cache-dir", $CacheDir,
    "--dtype", $DType,
    "--seed", "$Seed"
)
if ($Force) {
    $argsList += "--force"
}

Write-Host ("Building HU T2 Stage8c cache from {0} input file(s)" -f $inputFiles.Count)
python @argsList
