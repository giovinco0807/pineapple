param(
    [string]$RunName = ("regular-hu-t1-stage17-holdout-mc128-" + (Get-Date -Format "yyyyMMdd-HHmmss")),
    [string]$TargetInput = "outputs/training/hu_turn1_stage17_first_2k_mc32/split/holdout_mc32_baseline_annotated.jsonl",
    [string]$SplitSummary = "outputs/training/hu_turn1_stage17_first_2k_mc32/split/split_summary.json",
    [int]$VmCount = 100,
    [string]$MachineType = "e2-highcpu-4",
    [int]$BaseSeed = 2026071801,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $TargetInput)) {
    throw "Holdout target input not found: $TargetInput"
}
if (-not (Test-Path -LiteralPath $SplitSummary)) {
    throw "Split summary not found: $SplitSummary"
}

$summary = Get-Content -LiteralPath $SplitSummary -Raw | ConvertFrom-Json
$expectedTargets = [int]$summary.holdout_rows
$actualTargets = @([System.IO.File]::ReadLines((Resolve-Path -LiteralPath $TargetInput)) | Where-Object { $_.Trim() }).Count
if ($actualTargets -ne $expectedTargets) {
    throw "Holdout row mismatch: file=$actualTargets summary=$expectedTargets"
}
if ($expectedTargets -le 0) {
    throw "Holdout is empty"
}

$arguments = @{
    RunName = $RunName
    TargetInput = $TargetInput
    TotalTargets = $expectedTargets
    ShardTargets = 1
    BaseSeed = $BaseSeed
    SeedStride = 1000000
    VmCount = [Math]::Min($VmCount, $expectedTargets)
    MachineType = $MachineType
    FutureSamples = 128
    MaxActions = 0
    OpeningLookaheadSamples = 1
    Profile = "stage9f_p2"
    OpponentProfile = "stage9f_p2"
    SourceActionsOnly = $true
}
if ($DryRun) {
    $arguments.DryRun = $true
}
else {
    $arguments.CreateInstances = $true
}

& "$PSScriptRoot\Start-GcpHuTurn1RefinementRelabelRun.ps1" @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Start-GcpHuTurn1RefinementRelabelRun.ps1 failed with exit code $LASTEXITCODE"
}

if (-not $DryRun) {
    $runNamePath = "outputs/training/hu_turn1_stage17_first_2k_mc32/holdout_mc128_run_name.txt"
    New-Item -ItemType Directory -Force -Path (Split-Path $runNamePath -Parent) | Out-Null
    [System.IO.File]::WriteAllText(
        (Join-Path (Get-Location) $runNamePath),
        $RunName + [Environment]::NewLine,
        [System.Text.UTF8Encoding]::new($false)
    )
}
