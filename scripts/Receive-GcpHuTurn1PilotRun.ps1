param(
    [string]$ProjectId = "ofc-solver-485418",
    [string]$Bucket = "pokerhu-ofc-solver-485418-training",
    [Parameter(Mandatory = $true)]
    [string]$RunName,
    [string]$DownloadDir,
    [string]$OutputDir = "outputs/hu_turn1_stage1_pilot/gcp_aggregate",
    [string]$AggregatorModule = "ofc_regular.aggregate_hu_turn1_pilot",
    [string]$MergedOutputName = "hu_turn1_stage1_pilot.jsonl",
    [string]$ShardSummariesName = "hu_turn1_stage1_pilot_summaries.jsonl",
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
$inputDirs = @()
foreach ($spec in $specs) {
    $resultPath = Join-Path $resultsDir $spec.output_prefix
    $donePath = Join-Path $resultPath "DONE"
    if (-not (Test-Path $donePath)) {
        $missing += [int]$spec.shard
    }
    else {
        $inputDirs += $resultPath
    }
}
if ($missing.Count -gt 0 -and -not $AllowPartial) {
    throw "Missing HU T1 pilot result shards: $($missing -join ',')"
}
if ($missing.Count -gt 0) {
    Write-Warning "Aggregating partial HU T1 pilot results; missing shards: $($missing -join ',')"
}
if ($inputDirs.Count -eq 0) {
    throw "No completed HU T1 pilot result shards found"
}

$aggregateArgs = @(
    "-m", $AggregatorModule,
    "--results-root", "$resultsDir",
    "--shard-manifest", "$shardManifestPath",
    "--output-dir", "$OutputDir"
)
if ($AllowPartial) {
    $aggregateArgs += "--allow-partial"
}
python @aggregateArgs
if ($LASTEXITCODE -ne 0) {
    throw "HU T1 pilot aggregate failed with exit code $LASTEXITCODE"
}

[pscustomobject]@{
    run_name = $RunName
    project_id = $ProjectId
    bucket = $Bucket
    expected_shards = $expectedShards
    completed_shards = $inputDirs.Count
    missing_shards = $missing
    allow_partial = [bool]$AllowPartial
    output_dir = $OutputDir
    download_dir = $DownloadDir
    aggregate_summary = (Join-Path $OutputDir "summary.json")
    merged_output = (Join-Path $OutputDir $MergedOutputName)
    shard_summaries = (Join-Path $OutputDir $ShardSummariesName)
} | ConvertTo-Json -Depth 6
