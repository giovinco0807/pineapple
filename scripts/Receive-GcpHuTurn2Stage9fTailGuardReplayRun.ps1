param(
    [string]$ProjectId = "ofc-solver-485418",
    [string]$Bucket = "pokerhu-ofc-solver-485418-training",
    [Parameter(Mandatory = $true)]
    [string]$RunName,
    [string]$DownloadDir,
    [string]$OutputDir = "outputs/evals/hu_turn2_stage9f_tail_guard_replay_all_mc512_gcp",
    [switch]$AllowPartial,
    [switch]$SkipLabeling
)

$ErrorActionPreference = "Stop"

$cloudSdkGcloud = Join-Path $env:LOCALAPPDATA "Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
if (Test-Path $cloudSdkGcloud) {
    Set-Alias -Name gcloud -Value $cloudSdkGcloud -Scope Script
}
elseif (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    throw "gcloud not found in PATH or at $cloudSdkGcloud"
}

function Merge-CsvFiles {
    param([object[]]$Specs, [string]$ResultsDir, [string]$MergedPath, [string]$FileName)
    $rows = @()
    foreach ($spec in $Specs) {
        $path = Join-Path (Join-Path $ResultsDir $spec.output_prefix) $FileName
        if (Test-Path -LiteralPath $path) {
            $rows += Import-Csv -LiteralPath $path
        }
    }
    if ($rows.Count -gt 0) {
        $rows | Export-Csv -LiteralPath $MergedPath -NoTypeInformation -Encoding UTF8
    }
    else {
        "" | Set-Content -LiteralPath $MergedPath -Encoding UTF8
    }
    return $rows.Count
}

function Merge-JsonlFiles {
    param([object[]]$Specs, [string]$ResultsDir, [string]$MergedPath, [string]$FileName)
    if (Test-Path -LiteralPath $MergedPath) { Remove-Item -LiteralPath $MergedPath -Force }
    $count = 0
    foreach ($spec in $Specs) {
        $path = Join-Path (Join-Path $ResultsDir $spec.output_prefix) $FileName
        if (Test-Path -LiteralPath $path) {
            foreach ($line in [System.IO.File]::ReadLines((Resolve-Path -LiteralPath $path).Path)) {
                if (-not [string]::IsNullOrWhiteSpace($line)) {
                    Add-Content -LiteralPath $MergedPath -Value $line -Encoding UTF8
                    $count += 1
                }
            }
        }
    }
    return $count
}

if (-not $DownloadDir) {
    $DownloadDir = "outputs/gcp_runs/$RunName"
}

$prefix = "gs://$Bucket/runs/$RunName"
$manifestPath = Join-Path $DownloadDir "manifest.json"
$shardManifestPath = Join-Path $DownloadDir "shards_manifest.jsonl"
$resultsDir = Join-Path $DownloadDir "results"
$statusDir = Join-Path $DownloadDir "status"
$mergedDir = Join-Path $OutputDir "merged"
$labelsDir = Join-Path $OutputDir "labels"
New-Item -ItemType Directory -Force -Path $DownloadDir, $resultsDir, $statusDir, $OutputDir, $mergedDir | Out-Null

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
foreach ($line in [System.IO.File]::ReadLines((Resolve-Path -LiteralPath $shardManifestPath).Path)) {
    if ($line.Trim()) { $specs += ($line | ConvertFrom-Json) }
}
$expectedShards = [int]$manifest.total_shards
if ($specs.Count -ne $expectedShards) {
    throw "Shard manifest count mismatch: $($specs.Count) != $expectedShards"
}

$missing = @()
$completedSpecs = @()
foreach ($spec in $specs) {
    $resultPath = Join-Path $resultsDir $spec.output_prefix
    $donePath = Join-Path $resultPath "DONE"
    if (-not (Test-Path -LiteralPath $donePath)) {
        $missing += [int]$spec.shard
    }
    else {
        $completedSpecs += $spec
    }
}
if ($missing.Count -gt 0 -and -not $AllowPartial) {
    throw "Missing Stage9f tail-guard replay shards: $($missing -join ',')"
}
if ($missing.Count -gt 0) {
    Write-Warning "Merging partial Stage9f tail-guard replay; missing shards: $($missing -join ',')"
}
if ($completedSpecs.Count -eq 0) {
    throw "No completed Stage9f tail-guard replay shards found"
}

$mergedResultsCsv = Join-Path $mergedDir "replay_results.csv"
$mergedReadinessCsv = Join-Path $mergedDir "readiness.csv"
$mergedResultsJsonl = Join-Path $mergedDir "replay_results.jsonl"
$resultRows = Merge-CsvFiles -Specs $completedSpecs -ResultsDir $resultsDir -MergedPath $mergedResultsCsv -FileName "replay_results.csv"
$readinessRows = Merge-CsvFiles -Specs $completedSpecs -ResultsDir $resultsDir -MergedPath $mergedReadinessCsv -FileName "readiness.csv"
$jsonlRows = Merge-JsonlFiles -Specs $completedSpecs -ResultsDir $resultsDir -MergedPath $mergedResultsJsonl -FileName "replay_results.jsonl"

$labelManifest = $null
if (-not $SkipLabeling) {
    python -m ofc_regular.label_hu_turn2_stage9f_tail_guard_replay `
        --replay-results $mergedResultsCsv `
        --output-dir $labelsDir
    if ($LASTEXITCODE -ne 0) { throw "Stage9f tail-guard label generation failed with exit code $LASTEXITCODE" }
    $labelManifestPath = Join-Path $labelsDir "stage9f_tail_guard_label_manifest.json"
    if (Test-Path -LiteralPath $labelManifestPath) {
        $labelManifest = Get-Content -LiteralPath $labelManifestPath -Raw | ConvertFrom-Json
    }
}

[pscustomobject]@{
    run_name = $RunName
    project_id = $ProjectId
    bucket = $Bucket
    expected_shards = $expectedShards
    completed_shards = $completedSpecs.Count
    missing_shards = $missing
    allow_partial = [bool]$AllowPartial
    output_dir = $OutputDir
    download_dir = $DownloadDir
    merged_results_csv = $mergedResultsCsv
    merged_results_jsonl = $mergedResultsJsonl
    merged_readiness_csv = $mergedReadinessCsv
    result_rows = $resultRows
    result_jsonl_rows = $jsonlRows
    readiness_rows = $readinessRows
    labels_dir = if ($SkipLabeling) { "" } else { $labelsDir }
    label_manifest = $labelManifest
    production_p2_fixed = "No-Go"
    t1_training = "No-Go"
    teacher_50k = "No-Go"
} | ConvertTo-Json -Depth 8
