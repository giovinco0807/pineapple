param(
    [string]$ProjectId = "ofc-solver-485418",
    [string]$Bucket = "pokerhu-ofc-solver-485418-training",
    [Parameter(Mandatory = $true)]
    [string]$RunName,
    [int]$ExpectedShards = 0,
    [int]$ShardSamples = 0,
    [int]$ExpectedLines = 0,
    [string]$DownloadDir,
    [string]$MergedOutput
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    $cloudSdkGcloud = Join-Path $env:LOCALAPPDATA "Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
    if (-not (Test-Path $cloudSdkGcloud)) {
        throw "gcloud not found in PATH or at $cloudSdkGcloud"
    }
    Set-Alias -Name gcloud -Value $cloudSdkGcloud -Scope Script
}

function Get-LineCount {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        return 0
    }
    $reader = [System.IO.File]::OpenText((Resolve-Path $Path))
    try {
        $count = 0
        while ($null -ne $reader.ReadLine()) {
            $count += 1
        }
        return $count
    }
    finally {
        $reader.Close()
    }
}

if (-not $DownloadDir) {
    $DownloadDir = "outputs/gcp_runs/$RunName/shards"
}
if (-not $MergedOutput) {
    $MergedOutput = "outputs/gcp_runs/$RunName/turn3_teacher_merged.jsonl"
}

$prefix = "gs://$Bucket/runs/$RunName"
$manifestUri = "$prefix/manifest.json"
$manifest = $null
if ((gcloud storage ls $manifestUri --project $ProjectId 2>$null) -contains $manifestUri) {
    $manifestPath = Join-Path (Split-Path -Parent $DownloadDir) "manifest.json"
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $manifestPath) | Out-Null
    gcloud storage cp $manifestUri $manifestPath --project $ProjectId | Out-Null
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    if ($ExpectedShards -le 0 -and $manifest.total_shards) {
        $ExpectedShards = [int]$manifest.total_shards
    }
    if ($ShardSamples -le 0 -and $manifest.shard_samples) {
        $ShardSamples = [int]$manifest.shard_samples
    }
}

New-Item -ItemType Directory -Force -Path $DownloadDir | Out-Null
gcloud storage cp "$prefix/shards/*.jsonl" $DownloadDir --project $ProjectId | Out-Null

$files = Get-ChildItem -LiteralPath $DownloadDir -Filter "t3_*.jsonl" |
    Sort-Object Name

$shards = @()
foreach ($file in $files) {
    if ($file.Name -match '^t3_(\d+)\.jsonl$') {
        $lines = Get-LineCount $file.FullName
        $shards += [pscustomobject]@{
            shard = [int]$matches[1]
            path = $file.FullName
            lines = $lines
        }
    }
}
$shards = @($shards | Sort-Object shard)

$missing = @()
if ($ExpectedShards -gt 0) {
    $doneSet = @{}
    foreach ($shard in $shards) {
        $doneSet[$shard.shard] = $true
    }
    for ($i = 0; $i -lt $ExpectedShards; $i += 1) {
        if (-not $doneSet.ContainsKey($i)) {
            $missing += $i
        }
    }
    if ($missing.Count -gt 0) {
        throw "Missing shards: $($missing -join ',')"
    }
}

$badLineCounts = @()
if ($ShardSamples -gt 0) {
    foreach ($shard in $shards) {
        if ($shard.lines -ne $ShardSamples) {
            $badLineCounts += $shard
        }
    }
    if ($badLineCounts.Count -gt 0) {
        throw "Bad shard line counts: $($badLineCounts | ConvertTo-Json -Compress)"
    }
}

$mergedParent = Split-Path -Parent $MergedOutput
if ($mergedParent) {
    New-Item -ItemType Directory -Force -Path $mergedParent | Out-Null
}
$tempMerged = "$MergedOutput.tmp"
if (Test-Path $tempMerged) {
    Remove-Item -LiteralPath $tempMerged -Force
}

$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
$writer = [System.IO.StreamWriter]::new($tempMerged, $false, $utf8NoBom)
try {
    foreach ($shard in $shards) {
        foreach ($line in [System.IO.File]::ReadLines((Resolve-Path $shard.path))) {
            $writer.WriteLine($line)
        }
    }
}
finally {
    $writer.Close()
}
Move-Item -Force -Path $tempMerged -Destination $MergedOutput
$mergedLines = Get-LineCount $MergedOutput
if ($ExpectedLines -gt 0 -and $mergedLines -ne $ExpectedLines) {
    throw "Bad merged line count: expected $ExpectedLines, got $mergedLines"
}

[pscustomobject]@{
    run_name = $RunName
    project_id = $ProjectId
    bucket = $Bucket
    shards = $shards.Count
    lines = $mergedLines
    merged_output = $MergedOutput
    manifest = $manifest
} | ConvertTo-Json -Depth 5
