param(
    [string]$ProjectId = "ofc-solver-485418",
    [string]$Bucket = "pokerhu-ofc-solver-485418-training",
    [Parameter(Mandatory = $true)]
    [string]$RunName,
    [int]$MaxMissingToShow = 50
)

$ErrorActionPreference = "Stop"

$cloudSdkGcloud = Join-Path $env:LOCALAPPDATA "Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
if (Test-Path $cloudSdkGcloud) {
    Set-Alias -Name gcloud -Value $cloudSdkGcloud -Scope Script
}
elseif (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    throw "gcloud not found in PATH or at $cloudSdkGcloud"
}

$prefix = "gs://$Bucket/runs/$RunName"
$tmpRoot = Join-Path $env:TEMP ("{0}-hu-t2-stage9f-profile-canary-status" -f $RunName)
if (Test-Path $tmpRoot) { Remove-Item -LiteralPath $tmpRoot -Recurse -Force }
New-Item -ItemType Directory -Force -Path $tmpRoot | Out-Null

$manifest = $null
$manifestUri = "$prefix/manifest.json"
try {
    $manifestExists = gcloud storage ls $manifestUri --project $ProjectId 2>$null
}
catch {
    $manifestExists = @()
}
if ($manifestExists -contains $manifestUri) {
    $manifestPath = Join-Path $tmpRoot "manifest.json"
    gcloud storage cp $manifestUri $manifestPath --project $ProjectId | Out-Null
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
}

$expectedShards = if ($manifest -and $manifest.total_shards) { [int]$manifest.total_shards } else { 0 }
$statusLines = @()
$doneLines = @()
try {
    $doneList = gcloud storage ls "$prefix/results/*/DONE" --project $ProjectId 2>$null
    if ($doneList) { $doneLines = @($doneList | Where-Object { $_ -match '/DONE$' }) }
}
catch { $doneLines = @() }
try {
    $statusList = gcloud storage ls "$prefix/status/*.json" --project $ProjectId 2>$null
    if ($statusList) { $statusLines = @($statusList | Where-Object { $_ -match '\.json$' }) }
}
catch { $statusLines = @() }

$completed = @()
$failed = @()
$running = @()
if ($statusLines.Count -gt 0) {
    $statusDir = Join-Path $tmpRoot "status"
    New-Item -ItemType Directory -Force -Path $statusDir | Out-Null
    gcloud storage cp "$prefix/status/*.json" $statusDir --project $ProjectId | Out-Null
    foreach ($file in Get-ChildItem -LiteralPath $statusDir -Filter "*.json") {
        try {
            $status = Get-Content -LiteralPath $file.FullName -Raw | ConvertFrom-Json
            if ($status.status -eq "complete") { $completed += [int]$status.shard }
            elseif ($status.status -eq "failed") { $failed += [int]$status.shard }
            elseif ($status.status -eq "running") { $running += [int]$status.shard }
        }
        catch {
            $failed += -1
        }
    }
}
$completed = @($completed | Sort-Object -Unique)
$failed = @($failed | Sort-Object -Unique)
$running = @($running | Sort-Object -Unique)

$missing = @()
if ($expectedShards -gt 0) {
    $doneSet = @{}
    foreach ($n in $completed) { $doneSet[$n] = $true }
    for ($i = 0; $i -lt $expectedShards; $i += 1) {
        if (-not $doneSet.ContainsKey($i)) { $missing += $i }
    }
}

$vmPrefix = ($RunName.ToLowerInvariant() -replace '[^a-z0-9-]', '-').Trim('-')
if ($vmPrefix.Length -gt 54) { $vmPrefix = $vmPrefix.Substring(0, 54).Trim('-') }
$runningInstances = gcloud compute instances list `
    --project $ProjectId `
    --filter ("name~'^" + $vmPrefix + "' AND status=RUNNING") `
    --format "value(name,zone,status)" 2>$null

[pscustomobject]@{
    run_name = $RunName
    project_id = $ProjectId
    bucket = $Bucket
    expected_shards = $expectedShards
    completed_shards = $completed.Count
    running_shards = $running
    failed_shards = $failed
    done_files = $doneLines.Count
    status_files = $statusLines.Count
    missing_count = $missing.Count
    missing_shards = @($missing | Select-Object -First $MaxMissingToShow)
    running_instances = @($runningInstances)
    manifest = $manifest
} | ConvertTo-Json -Depth 8
