param(
    [string]$ProjectId = "ofc-solver-485418",
    [string]$Bucket = "pokerhu-ofc-solver-485418-training",
    [Parameter(Mandatory = $true)]
    [string]$RunName,
    [string]$DownloadDir,
    [string]$OutputDir,
    [string]$MergedOutputName = "hu_turn2_pilot_2000_mc512.jsonl"
)

$ErrorActionPreference = "Stop"

$cloudSdkGcloud = Join-Path $env:LOCALAPPDATA "Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
if (Test-Path $cloudSdkGcloud) {
    Set-Alias -Name gcloud -Value $cloudSdkGcloud -Scope Script
}
elseif (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    throw "gcloud not found in PATH or at $cloudSdkGcloud"
}

function Get-LineCount {
    param([string]$Path)
    if (-not (Test-Path $Path)) { return 0 }
    $reader = [System.IO.File]::OpenText((Resolve-Path $Path))
    try {
        $count = 0
        while ($null -ne $reader.ReadLine()) { $count += 1 }
        return $count
    }
    finally {
        $reader.Close()
    }
}

if (-not $OutputDir) {
    $OutputDir = "outputs/hu_turn2_stage1_batch9_3_pilot_2000_mc512"
}
if (-not $DownloadDir) {
    $DownloadDir = "outputs/gcp_runs/$RunName"
}

$prefix = "gs://$Bucket/runs/$RunName"
$manifestPath = Join-Path $DownloadDir "manifest.json"
$shardManifestPath = Join-Path $DownloadDir "shards_manifest.jsonl"
$shardDir = Join-Path $DownloadDir "shards"
$summaryDir = Join-Path $DownloadDir "summaries"
New-Item -ItemType Directory -Force -Path $DownloadDir, $shardDir, $summaryDir, $OutputDir | Out-Null

gcloud storage cp "$prefix/manifest.json" $manifestPath --project $ProjectId | Out-Null
gcloud storage cp "$prefix/source/shards_manifest.jsonl" $shardManifestPath --project $ProjectId | Out-Null
gcloud storage cp "$prefix/shards/*.jsonl" $shardDir --project $ProjectId | Out-Null
try {
    gcloud storage cp "$prefix/summaries/*.json" $summaryDir --project $ProjectId | Out-Null
}
catch {
    Write-Warning "No summaries downloaded or summary download failed: $_"
}

$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$specs = @()
foreach ($line in [System.IO.File]::ReadLines((Resolve-Path $shardManifestPath))) {
    if ($line.Trim()) {
        $specs += ($line | ConvertFrom-Json)
    }
}
$expectedShards = [int]$manifest.total_shards
if ($specs.Count -ne $expectedShards) {
    throw "Shard manifest count mismatch: $($specs.Count) != $expectedShards"
}

$missing = @()
$badCounts = @()
foreach ($spec in $specs) {
    $path = Join-Path $shardDir $spec.output_name
    if (-not (Test-Path $path)) {
        $missing += [int]$spec.shard
        continue
    }
    $lines = Get-LineCount $path
    if ($lines -ne [int]$spec.samples) {
        $badCounts += [pscustomobject]@{ shard = [int]$spec.shard; path = $path; lines = $lines; expected = [int]$spec.samples }
    }
}
if ($missing.Count -gt 0) {
    throw "Missing shards: $($missing -join ',')"
}
if ($badCounts.Count -gt 0) {
    throw "Bad shard line counts: $($badCounts | ConvertTo-Json -Compress)"
}

$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
$mergedOutput = Join-Path $OutputDir $MergedOutputName
$bucketWriters = @{}
$mergedWriter = [System.IO.StreamWriter]::new("$mergedOutput.tmp", $false, $utf8NoBom)
try {
    foreach ($bucket in @($specs.bucket | Sort-Object -Unique)) {
        $bucketPath = Join-Path $OutputDir ("{0}.jsonl" -f $bucket)
        $bucketWriters[$bucket] = [System.IO.StreamWriter]::new("$bucketPath.tmp", $false, $utf8NoBom)
    }
    foreach ($spec in ($specs | Sort-Object shard)) {
        $path = Join-Path $shardDir $spec.output_name
        foreach ($line in [System.IO.File]::ReadLines((Resolve-Path $path))) {
            $mergedWriter.WriteLine($line)
            $bucketWriters[$spec.bucket].WriteLine($line)
        }
    }
}
finally {
    $mergedWriter.Close()
    foreach ($writer in $bucketWriters.Values) { $writer.Close() }
}
Move-Item -Force -Path "$mergedOutput.tmp" -Destination $mergedOutput
foreach ($bucket in $bucketWriters.Keys) {
    $bucketPath = Join-Path $OutputDir ("{0}.jsonl" -f $bucket)
    Move-Item -Force -Path "$bucketPath.tmp" -Destination $bucketPath
}

$bucketCounts = @()
foreach ($bucket in @($specs.bucket | Sort-Object -Unique)) {
    $bucketPath = Join-Path $OutputDir ("{0}.jsonl" -f $bucket)
    $bucketCounts += [pscustomobject]@{
        bucket = $bucket
        path = $bucketPath
        lines = Get-LineCount $bucketPath
    }
}

[pscustomobject]@{
    run_name = $RunName
    project_id = $ProjectId
    bucket = $Bucket
    expected_shards = $expectedShards
    merged_output = $mergedOutput
    merged_lines = Get-LineCount $mergedOutput
    output_dir = $OutputDir
    download_dir = $DownloadDir
    bucket_counts = $bucketCounts
} | ConvertTo-Json -Depth 5
