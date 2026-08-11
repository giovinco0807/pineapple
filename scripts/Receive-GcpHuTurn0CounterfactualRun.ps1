param(
    [string]$ProjectId = "ofc-solver-485418",
    [string]$Bucket = "pokerhu-ofc-solver-485418-training",
    [Parameter(Mandatory = $true)]
    [string]$RunName,
    [string]$DownloadDir,
    [string]$OutputDir = "outputs/evals/hu_turn0_counterfactual_gcp",
    [switch]$AllowPartial
)

$ErrorActionPreference = "Stop"
if (-not $DownloadDir) { $DownloadDir = "outputs/gcp_runs/$RunName" }
$prefix = "gs://$Bucket/runs/$RunName"
$manifestPath = Join-Path $DownloadDir "manifest.json"
$shardManifestPath = Join-Path $DownloadDir "shards_manifest.jsonl"
$resultsDir = Join-Path $DownloadDir "results"
$statusDir = Join-Path $DownloadDir "status"
New-Item -ItemType Directory -Force -Path $DownloadDir,$resultsDir,$statusDir,$OutputDir | Out-Null
gcloud storage cp "$prefix/manifest.json" $manifestPath --project $ProjectId | Out-Null
gcloud storage cp "$prefix/source/shards_manifest.jsonl" $shardManifestPath --project $ProjectId | Out-Null
gcloud storage cp --recursive "$prefix/results/*" $resultsDir --project $ProjectId | Out-Null
try { gcloud storage cp "$prefix/status/*.json" $statusDir --project $ProjectId | Out-Null }
catch { Write-Warning "Status download failed: $_" }

$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$missing = @()
$completed = @()
foreach ($line in [System.IO.File]::ReadLines((Resolve-Path $shardManifestPath))) {
    if (-not $line.Trim()) { continue }
    $spec = $line | ConvertFrom-Json
    $resultDir = Join-Path $resultsDir ([string]$spec.output_prefix)
    if (Test-Path (Join-Path $resultDir "DONE")) { $completed += [int]$spec.shard }
    else { $missing += [int]$spec.shard }
}
if ($missing.Count -gt 0 -and -not $AllowPartial) {
    throw "Missing HU T0 counterfactual shards: $($missing -join ',')"
}
if ($completed.Count -eq 0) { throw "No completed HU T0 counterfactual shards" }
$aggregateArgs = @(
    "-m","ofc_regular.aggregate_hu_turn0_counterfactual",
    "--results-root",$resultsDir,
    "--output-dir",$OutputDir
)
if (-not $AllowPartial) {
    $aggregateArgs += @("--expected-paired-seeds-per-config",[string]$manifest.paired_seeds_per_config)
}
& python @aggregateArgs
if ($LASTEXITCODE -ne 0) { throw "HU T0 counterfactual aggregation failed" }
[pscustomobject]@{
    schema = "hu_turn0_counterfactual_gcp_receive_v1"
    run_name = $RunName
    expected_shards = [int]$manifest.total_shards
    completed_shards = $completed.Count
    missing_shards = $missing
    allow_partial = [bool]$AllowPartial
    output_dir = $OutputDir
    summary = (Join-Path $OutputDir "summary.json")
    events = (Join-Path $OutputDir "events_merged.jsonl")
} | ConvertTo-Json -Depth 6
