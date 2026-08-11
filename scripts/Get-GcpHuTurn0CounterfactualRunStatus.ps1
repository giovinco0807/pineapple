param(
    [string]$ProjectId = "ofc-solver-485418",
    [string]$Bucket = "pokerhu-ofc-solver-485418-training",
    [Parameter(Mandatory = $true)]
    [string]$RunName,
    [int]$MaxMissingToShow = 30
)

$ErrorActionPreference = "Stop"
$tempDir = Join-Path $env:TEMP "$RunName-hu-t0-counterfactual-status"
New-Item -ItemType Directory -Force -Path $tempDir | Out-Null
$manifestPath = Join-Path $tempDir "manifest.json"
$shardManifestPath = Join-Path $tempDir "shards_manifest.jsonl"
$prefix = "gs://$Bucket/runs/$RunName"
gcloud storage cp "$prefix/manifest.json" $manifestPath --project $ProjectId | Out-Null
gcloud storage cp "$prefix/source/shards_manifest.jsonl" $shardManifestPath --project $ProjectId | Out-Null
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$doneUrls = @(gcloud storage ls "$prefix/results/*/DONE" --project $ProjectId 2>$null)
$donePrefixes = New-Object 'System.Collections.Generic.HashSet[string]'
foreach ($url in $doneUrls) {
    if ($url -match '/results/([^/]+)/DONE$') { [void]$donePrefixes.Add($Matches[1]) }
}
$missing = @()
foreach ($line in [System.IO.File]::ReadLines((Resolve-Path $shardManifestPath))) {
    if (-not $line.Trim()) { continue }
    $spec = $line | ConvertFrom-Json
    if (-not $donePrefixes.Contains([string]$spec.output_prefix)) { $missing += [int]$spec.shard }
}
$vmPrefix = (($RunName.ToLowerInvariant() -replace '[^a-z0-9-]', '-').Trim('-'))
if ($vmPrefix.Length -gt 54) { $vmPrefix = $vmPrefix.Substring(0,54).Trim('-') }
$running = @(gcloud compute instances list --project $ProjectId --filter "status=RUNNING AND name~'$vmPrefix'" --format "value(name)" 2>$null)
[pscustomobject]@{
    schema = "hu_turn0_counterfactual_gcp_status_v1"
    run_name = $RunName
    expected_shards = [int]$manifest.total_shards
    completed_shards = $donePrefixes.Count
    missing_shards = $missing.Count
    first_missing = @($missing | Select-Object -First $MaxMissingToShow)
    running_instances = $running.Count
    paired_seeds_per_config = [int]$manifest.paired_seeds_per_config
    configs = @($manifest.configs).Count
} | ConvertTo-Json -Depth 6
