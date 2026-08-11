param(
    [string]$ProjectId = "ofc-solver-485418",
    [string]$Bucket = "pokerhu-ofc-solver-485418-training",
    [Parameter(Mandatory = $true)]
    [string]$RunName,
    [string]$DownloadDir,
    [string]$OutputDir = "outputs/hu_turn1_stage1j_stratified_teacher/gcp_refinement_relabel",
    [switch]$AllowPartial
)

$ErrorActionPreference = "Stop"

$cloudSdkGcloud = Join-Path $env:LOCALAPPDATA "Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
if (Test-Path $cloudSdkGcloud) {
    Set-Alias -Name gcloud -Value $cloudSdkGcloud -Scope Script
}
elseif (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    throw "gcloud not found in PATH or at $cloudSdkGcloud"
}

if (-not $DownloadDir) {
    $DownloadDir = "outputs/gcp_runs/$RunName"
}

$prefix = "gs://$Bucket/runs/$RunName"
$manifestPath = Join-Path $DownloadDir "manifest.json"
$shardManifestPath = Join-Path $DownloadDir "shards_manifest.jsonl"
$resultsDir = Join-Path $DownloadDir "results"
$statusDir = Join-Path $DownloadDir "status"
New-Item -ItemType Directory -Force -Path $DownloadDir, $resultsDir, $statusDir, $OutputDir | Out-Null

gcloud storage cp "$prefix/manifest.json" $manifestPath --project $ProjectId | Out-Null
gcloud storage cp "$prefix/source/shards_manifest.jsonl" $shardManifestPath --project $ProjectId | Out-Null
gcloud storage cp --recursive "$prefix/results/*" $resultsDir --project $ProjectId | Out-Null
try {
    gcloud storage cp "$prefix/status/*.json" $statusDir --project $ProjectId | Out-Null
}
catch {
    Write-Warning "No statuses downloaded or status download failed: $_"
}

$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$specs = @()
foreach ($line in [System.IO.File]::ReadLines((Resolve-Path $shardManifestPath))) {
    if ($line.Trim()) { $specs += ($line | ConvertFrom-Json) }
}
$expectedShards = [int]$manifest.total_shards
if ($specs.Count -ne $expectedShards) {
    throw "Shard manifest count mismatch: $($specs.Count) != $expectedShards"
}

$missing = @()
$completed = @()
foreach ($spec in $specs) {
    $resultPath = Join-Path $resultsDir $spec.output_prefix
    $donePath = Join-Path $resultPath "DONE"
    $jsonlPath = Join-Path $resultPath "relabel.jsonl"
    $summaryPath = Join-Path $resultPath "summary.json"
    if (-not (Test-Path $donePath) -or -not (Test-Path $jsonlPath) -or -not (Test-Path $summaryPath)) {
        $missing += [int]$spec.shard
    }
    else {
        $completed += $spec
    }
}
if ($missing.Count -gt 0 -and -not $AllowPartial) {
    throw "Missing HU T1 refinement relabel result shards: $($missing -join ',')"
}
if ($missing.Count -gt 0) {
    Write-Warning "Aggregating partial HU T1 refinement relabel results; missing shards: $($missing -join ',')"
}
if ($completed.Count -eq 0) {
    throw "No completed HU T1 refinement relabel result shards found"
}

$mergedOutput = Join-Path $OutputDir "hu_turn1_refinement_relabel.jsonl"
$summariesOutput = Join-Path $OutputDir "hu_turn1_refinement_relabel_summaries.jsonl"
$summaryOutput = Join-Path $OutputDir "summary.json"
Remove-Item -Force -ErrorAction SilentlyContinue $mergedOutput, $summariesOutput, $summaryOutput

$totalRows = 0
$summaryRows = @()
foreach ($spec in ($completed | Sort-Object {[int]$_.shard})) {
    $resultPath = Join-Path $resultsDir $spec.output_prefix
    $jsonlPath = Join-Path $resultPath "relabel.jsonl"
    $summaryPath = Join-Path $resultPath "summary.json"
    Get-Content -LiteralPath $jsonlPath | Add-Content -Encoding UTF8 $mergedOutput
    $lineCount = (Get-Content -LiteralPath $jsonlPath | Measure-Object -Line).Lines
    $totalRows += $lineCount
    $summary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
    $summaryRows += $summary
    $summary | ConvertTo-Json -Compress -Depth 8 | Add-Content -Encoding UTF8 $summariesOutput
}

$changed = 0
$relabelSeconds = 0.0
$t2Seconds = 0.0
$targetAttempts = 0
$skippedTargets = 0
$timedOutTargets = 0
$failedTargets = 0
$seatCounts = @{}
$reasonCounts = @{}
$skipReasonCounts = @{}
foreach ($summary in $summaryRows) {
    $changedValue = 0
    $relabelValue = 0.0
    $t2Value = 0.0
    $attemptValue = 0
    $skippedValue = 0
    $timedOutValue = 0
    $failedValue = 0
    if ($summary.PSObject.Properties.Name -contains "best_action_changed") {
        $changedValue = [int]$summary.best_action_changed
    }
    if ($summary.PSObject.Properties.Name -contains "relabel_seconds_sum") {
        $relabelValue = [double]$summary.relabel_seconds_sum
    }
    if ($summary.PSObject.Properties.Name -contains "t2_choose_action_seconds_sum") {
        $t2Value = [double]$summary.t2_choose_action_seconds_sum
    }
    if ($summary.PSObject.Properties.Name -contains "target_attempts") {
        $attemptValue = [int]$summary.target_attempts
    }
    else {
        $attemptValue = [int]$summary.records
    }
    if ($summary.PSObject.Properties.Name -contains "skipped_targets") {
        $skippedValue = [int]$summary.skipped_targets
    }
    if ($summary.PSObject.Properties.Name -contains "timed_out_targets") {
        $timedOutValue = [int]$summary.timed_out_targets
    }
    if ($summary.PSObject.Properties.Name -contains "failed_targets") {
        $failedValue = [int]$summary.failed_targets
    }
    $changed += $changedValue
    $relabelSeconds += $relabelValue
    $t2Seconds += $t2Value
    $targetAttempts += $attemptValue
    $skippedTargets += $skippedValue
    $timedOutTargets += $timedOutValue
    $failedTargets += $failedValue
    foreach ($prop in ($summary.seat_counts.PSObject.Properties)) {
        $previous = 0
        if ($seatCounts.ContainsKey($prop.Name)) { $previous = [int]$seatCounts[$prop.Name] }
        $seatCounts[$prop.Name] = $previous + [int]$prop.Value
    }
    foreach ($prop in ($summary.reason_counts.PSObject.Properties)) {
        $previous = 0
        if ($reasonCounts.ContainsKey($prop.Name)) { $previous = [int]$reasonCounts[$prop.Name] }
        $reasonCounts[$prop.Name] = $previous + [int]$prop.Value
    }
    if ($summary.PSObject.Properties.Name -contains "skip_reason_counts") {
        foreach ($prop in ($summary.skip_reason_counts.PSObject.Properties)) {
            $previous = 0
            if ($skipReasonCounts.ContainsKey($prop.Name)) { $previous = [int]$skipReasonCounts[$prop.Name] }
            $skipReasonCounts[$prop.Name] = $previous + [int]$prop.Value
        }
    }
}

$aggregate = [ordered]@{
    schema = "hu_turn1_stage1_refinement_relabel_gcp_aggregate_v1"
    run_name = $RunName
    project_id = $ProjectId
    bucket = $Bucket
    manifest = $manifest
    expected_shards = $expectedShards
    completed_shards = $completed.Count
    missing_shards = $missing
    allow_partial = [bool]$AllowPartial
    records = $totalRows
    target_attempts = $targetAttempts
    skipped_targets = $skippedTargets
    timed_out_targets = $timedOutTargets
    failed_targets = $failedTargets
    best_action_changed = $changed
    best_action_changed_rate = if ($totalRows -gt 0) { $changed / $totalRows } else { 0.0 }
    relabel_seconds_sum = $relabelSeconds
    t2_choose_action_seconds_sum = $t2Seconds
    mean_relabel_seconds_per_record = if ($totalRows -gt 0) { $relabelSeconds / $totalRows } else { 0.0 }
    seat_counts = $seatCounts
    reason_counts = $reasonCounts
    skip_reason_counts = $skipReasonCounts
    merged_output = $mergedOutput
    shard_summaries = $summariesOutput
    download_dir = $DownloadDir
    output_dir = $OutputDir
    created_at = (Get-Date).ToUniversalTime().ToString("o")
}
$aggregate | ConvertTo-Json -Depth 10 | Set-Content -Encoding UTF8 $summaryOutput
$aggregate | ConvertTo-Json -Depth 10
