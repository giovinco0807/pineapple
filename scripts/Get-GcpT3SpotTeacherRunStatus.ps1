param(
    [string]$ProjectId = "ofc-solver-485418",
    [string]$Bucket = "pokerhu-ofc-solver-485418-training",
    [Parameter(Mandatory = $true)]
    [string]$RunName,
    [int]$ExpectedShards = 0,
    [int]$MaxMissingToShow = 50
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    $cloudSdkGcloud = Join-Path $env:LOCALAPPDATA "Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
    if (-not (Test-Path $cloudSdkGcloud)) {
        throw "gcloud not found in PATH or at $cloudSdkGcloud"
    }
    Set-Alias -Name gcloud -Value $cloudSdkGcloud -Scope Script
}

$prefix = "gs://$Bucket/runs/$RunName"
$manifestUri = "$prefix/manifest.json"
$manifest = $null
if ((gcloud storage ls $manifestUri --project $ProjectId 2>$null) -contains $manifestUri) {
    $tmpManifest = Join-Path $env:TEMP ("{0}-manifest.json" -f $RunName)
    gcloud storage cp $manifestUri $tmpManifest --project $ProjectId | Out-Null
    $manifest = Get-Content -LiteralPath $tmpManifest -Raw | ConvertFrom-Json
    if ($ExpectedShards -le 0 -and $manifest.total_shards) {
        $ExpectedShards = [int]$manifest.total_shards
    }
}

$shardLines = @()
$statusLines = @()
try {
    $shardList = gcloud storage ls "$prefix/shards/*.jsonl" --project $ProjectId 2>$null
    if ($shardList) {
        $shardLines = @($shardList | Where-Object { $_ -match '\.jsonl$' })
    }
}
catch {
    $shardLines = @()
}
try {
    $statusList = gcloud storage ls "$prefix/status/*.json" --project $ProjectId 2>$null
    if ($statusList) {
        $statusLines = @($statusList | Where-Object { $_ -match '\.json$' })
    }
}
catch {
    $statusLines = @()
}

$completedShardNumbers = @()
foreach ($line in $shardLines) {
    if ($line -match 't3_(\d+)\.jsonl$') {
        $completedShardNumbers += [int]$matches[1]
    }
}
$completedShardNumbers = @($completedShardNumbers | Sort-Object -Unique)

$missing = @()
if ($ExpectedShards -gt 0) {
    $doneSet = @{}
    foreach ($n in $completedShardNumbers) {
        $doneSet[$n] = $true
    }
    for ($i = 0; $i -lt $ExpectedShards; $i += 1) {
        if (-not $doneSet.ContainsKey($i)) {
            $missing += $i
        }
    }
}

$runningInstances = gcloud compute instances list `
    --project $ProjectId `
    --filter ("name~'^" + ($RunName.ToLowerInvariant() -replace '[^a-z0-9-]', '-') + "' AND status=RUNNING") `
    --format "value(name,zone,status)" 2>$null

$runningStartShards = @()
foreach ($line in @($runningInstances)) {
    $name = ($line -split "\s+")[0]
    if ($name -match '-(\d{3})$') {
        $runningStartShards += [int]$matches[1]
    }
}
$runningStartShards = @($runningStartShards | Sort-Object -Unique)

$missingStartShards = @()
if ($ExpectedShards -gt 0 -and $manifest -and $manifest.vm_count) {
    $stride = [int]$manifest.vm_count
    foreach ($missingShard in $missing) {
        $missingStartShards += ($missingShard % $stride)
    }
}
$missingStartShards = @($missingStartShards | Sort-Object -Unique)
$inactiveMissingStartShards = @()
foreach ($startShard in $missingStartShards) {
    if ($runningStartShards -notcontains $startShard) {
        $inactiveMissingStartShards += $startShard
    }
}

[pscustomobject]@{
    run_name = $RunName
    project_id = $ProjectId
    bucket = $Bucket
    expected_shards = $ExpectedShards
    completed_shards = $completedShardNumbers.Count
    status_files = @($statusLines).Count
    missing_count = @($missing).Count
    missing_shards = @($missing | Select-Object -First $MaxMissingToShow)
    missing_start_shards = $missingStartShards
    running_start_shards = $runningStartShards
    inactive_missing_start_shards = $inactiveMissingStartShards
    running_instances = @($runningInstances)
    manifest = $manifest
} | ConvertTo-Json -Depth 6
