param(
    [string]$ProjectId = "ofc-solver-485418",
    [string]$Bucket = "pokerhu-ofc-solver-485418-training",
    [string]$RunName = "regular-hu-t1-stage17-first2k-allaction-mc32-20260710-001",
    [string]$OutputRoot = "outputs/training/hu_turn1_stage17_first_2k_mc32",
    [int]$ExpectedShards = 400,
    [int]$ExpectedRecords = 2000,
    [int]$FutureSamples = 32,
    [double]$HoldoutFraction = 0.10,
    [int]$SplitSeed = 2026071704,
    [int]$BaselineSeed = 2026071705
)

$ErrorActionPreference = "Stop"

$cloudSdkGcloud = Join-Path $env:LOCALAPPDATA "Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
if (Test-Path $cloudSdkGcloud) {
    Set-Alias -Name gcloud -Value $cloudSdkGcloud -Scope Script
}
elseif (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    throw "gcloud not found in PATH or at $cloudSdkGcloud"
}

function Invoke-PythonChecked {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "python $($Arguments -join ' ') failed with exit code $LASTEXITCODE"
    }
}

$prefix = "gs://$Bucket/runs/$RunName"
$doneCount = @(gcloud storage ls "$prefix/results/**/DONE" --project $ProjectId 2>$null).Count
if ($doneCount -ne $ExpectedShards) {
    throw "Stage17 is incomplete: $doneCount / $ExpectedShards DONE shards"
}

$aggregateDir = Join-Path $OutputRoot "final_aggregate"
$analysisDir = Join-Path $OutputRoot "final_analysis"
$splitDir = Join-Path $OutputRoot "split"
$downloadDir = "outputs/gcp_runs/$RunName"
New-Item -ItemType Directory -Force -Path $aggregateDir, $analysisDir, $splitDir | Out-Null

& "$PSScriptRoot\Receive-GcpHuTurn1PilotRun.ps1" `
    -ProjectId $ProjectId `
    -Bucket $Bucket `
    -RunName $RunName `
    -DownloadDir $downloadDir `
    -OutputDir $aggregateDir
if ($LASTEXITCODE -ne 0) {
    throw "Receive-GcpHuTurn1PilotRun.ps1 failed with exit code $LASTEXITCODE"
}

$teacherPath = Join-Path $aggregateDir "hu_turn1_stage1_pilot.jsonl"
$auditPath = Join-Path $aggregateDir "audit.json"
Invoke-PythonChecked -Arguments @(
    "-m", "ofc_regular.audit_hu_turn1_teacher",
    "--input", $teacherPath,
    "--output", $auditPath,
    "--expected-records", "$ExpectedRecords",
    "--expected-seat", "first",
    "--future-samples", "$FutureSamples",
    "--profile", "stage9f_p2",
    "--opponent-profile", "stage9f_p2",
    "--t3-continuation", "stage7_m5_r10"
)

Invoke-PythonChecked -Arguments @(
    "-m", "ofc_regular.analyze_hu_turn1_pilot",
    "--input", $teacherPath,
    "--output-dir", $analysisDir,
    "--summary-output", (Join-Path $analysisDir "summary.json")
)

$trainPath = Join-Path $splitDir "train_mc32.jsonl"
$holdoutPath = Join-Path $splitDir "holdout_mc32.jsonl"
$splitSummaryPath = Join-Path $splitDir "split_summary.json"
Invoke-PythonChecked -Arguments @(
    "-m", "ofc_regular.split_hu_turn1_teacher",
    "--input", $teacherPath,
    "--train-output", $trainPath,
    "--holdout-output", $holdoutPath,
    "--summary-output", $splitSummaryPath,
    "--holdout-fraction", "$HoldoutFraction",
    "--seed", "$SplitSeed"
)

$trainAnnotatedPath = Join-Path $splitDir "train_mc32_baseline_annotated.jsonl"
$holdoutAnnotatedPath = Join-Path $splitDir "holdout_mc32_baseline_annotated.jsonl"
Invoke-PythonChecked -Arguments @(
    "-m", "ofc_regular.annotate_hu_turn1_teacher_baseline",
    "--input", $trainPath,
    "--output", $trainAnnotatedPath,
    "--summary-output", (Join-Path $splitDir "train_baseline_annotation_summary.json"),
    "--profile", "stage9f_p2",
    "--seed", "$BaselineSeed",
    "--opening-lookahead-samples", "1"
)
Invoke-PythonChecked -Arguments @(
    "-m", "ofc_regular.annotate_hu_turn1_teacher_baseline",
    "--input", $holdoutPath,
    "--output", $holdoutAnnotatedPath,
    "--summary-output", (Join-Path $splitDir "holdout_baseline_annotation_summary.json"),
    "--profile", "stage9f_p2",
    "--seed", "$BaselineSeed",
    "--opening-lookahead-samples", "1"
)

$splitSummary = Get-Content -LiteralPath $splitSummaryPath -Raw | ConvertFrom-Json
$audit = Get-Content -LiteralPath $auditPath -Raw | ConvertFrom-Json
[pscustomobject]@{
    status = "complete"
    run_name = $RunName
    completed_shards = $doneCount
    teacher_records = $audit.records
    teacher_actions = $audit.total_actions
    audit_status = $audit.status
    train_records = $splitSummary.train_rows
    holdout_records = $splitSummary.holdout_rows
    teacher = $teacherPath
    audit = $auditPath
    train = $trainAnnotatedPath
    holdout = $holdoutAnnotatedPath
    next_step = "Relabel the held-out rows with all-action common-future MC128 on GCP Spot."
} | ConvertTo-Json -Depth 5
