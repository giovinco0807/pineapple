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

function Get-Sha256 {
    param([string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

$prefix = "gs://$Bucket/runs/$RunName"
$safeName = ($RunName -replace '[^a-zA-Z0-9._-]', '-')
$tmpRoot = Join-Path $env:TEMP ("{0}-hu-m42-status-{1}" -f $safeName, $PID)
New-Item -ItemType Directory -Force -Path $tmpRoot | Out-Null

try {
    $manifestPath = Join-Path $tmpRoot "manifest.json"
    $shardsPath = Join-Path $tmpRoot "shards_manifest.jsonl"
    gcloud storage cp "$prefix/manifest.json" $manifestPath --project $ProjectId | Out-Null
    gcloud storage cp "$prefix/source/shards_manifest.jsonl" $shardsPath --project $ProjectId | Out-Null
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    if ($manifest.schema -ne "hu_m42_spot_manifest_v1") {
        throw "M4.2 manifest schema mismatch"
    }
    $actualShardsSha256 = Get-Sha256 $shardsPath
    $actualManifestSha256 = Get-Sha256 $manifestPath
    if ($actualShardsSha256 -ne $manifest.shards_manifest_sha256) {
        throw "M4.2 shard manifest SHA256 mismatch"
    }

    $specs = @()
    foreach ($line in [System.IO.File]::ReadLines((Resolve-Path $shardsPath))) {
        if ($line.Trim()) { $specs += ($line | ConvertFrom-Json) }
    }
    if ($specs.Count -ne [int]$manifest.total_shards) {
        throw "M4.2 shard manifest count mismatch"
    }
    $specByPrefix = @{}
    foreach ($spec in $specs) { $specByPrefix[[string]$spec.output_prefix] = $spec }

    $doneUris = @()
    try {
        $listed = gcloud storage ls "$prefix/results/*/DONE" --project $ProjectId 2>$null
        if ($listed) { $doneUris = @($listed | Where-Object { $_ -match '/DONE$' }) }
    }
    catch { $doneUris = @() }

    $doneRecords = @()
    foreach ($uri in $doneUris) {
        $outputPrefix = (($uri -split '/')[-2])
        if (-not $specByPrefix.ContainsKey($outputPrefix)) { continue }
        $spec = $specByPrefix[$outputPrefix]
        $destination = Join-Path $tmpRoot ("done_{0:D3}.json" -f [int]$spec.shard)
        gcloud storage cp $uri $destination --project $ProjectId | Out-Null
        try {
            $done = Get-Content -LiteralPath $destination -Raw | ConvertFrom-Json
            if ($done.schema -eq "hu_m42_spot_done_v1" -and
                $done.status -eq "complete" -and
                [int]$done.shard -eq [int]$spec.shard -and
                [string]$done.split -eq [string]$spec.split -and
                [int]$done.roots -eq [int]$spec.roots -and
                [string]$done.output_prefix -eq $outputPrefix -and
                [string]$done.manifest_sha256 -eq $actualManifestSha256 -and
                [string]$done.source_sha256 -eq [string]$manifest.source_sha256 -and
                [string]$done.startup_sha256 -eq [string]$manifest.startup_sha256 -and
                [string]$done.shards_manifest_sha256 -eq $actualShardsSha256 -and
                [string]$done.model_manifest_sha256 -eq [string]$manifest.model_manifest_sha256 -and
                [string]$done.native_manifest_sha256 -eq [string]$manifest.native_manifest_sha256) {
                $doneRecords += $done
            }
            else {
                throw "Stale or mismatched M4.2 DONE object: $uri"
            }
        }
        catch {
            throw "Invalid or stale M4.2 DONE object: $uri. $($_.Exception.Message)"
        }
    }
    $completedShards = @($doneRecords | ForEach-Object { [int]$_.shard } | Sort-Object -Unique)

    $statusRecords = @()
    try {
        $statusUris = @(gcloud storage ls "$prefix/status/shard_*.json" --project $ProjectId 2>$null)
        foreach ($uri in $statusUris) {
            if ($uri -notmatch 'shard_(\d+)\.json$') { continue }
            $destination = Join-Path $tmpRoot ("status_{0:D3}.json" -f [int]$matches[1])
            gcloud storage cp $uri $destination --project $ProjectId | Out-Null
            try { $statusRecords += (Get-Content -LiteralPath $destination -Raw | ConvertFrom-Json) }
            catch { Write-Warning "Invalid status object: $uri" }
        }
    }
    catch { $statusRecords = @() }

    $progress = @()
    try {
        $heartbeatUris = @(gcloud storage ls "$prefix/resume/*/heartbeat.json" --project $ProjectId 2>$null)
        foreach ($uri in $heartbeatUris) {
            if ($uri -notmatch '/resume/([^/]+)/heartbeat\.json$') { continue }
            $outputPrefix = $matches[1]
            if (-not $specByPrefix.ContainsKey($outputPrefix)) { continue }
            $spec = $specByPrefix[$outputPrefix]
            $destination = Join-Path $tmpRoot ("heartbeat_{0:D3}.json" -f [int]$spec.shard)
            gcloud storage cp $uri $destination --project $ProjectId | Out-Null
            try {
                $heartbeat = Get-Content -LiteralPath $destination -Raw | ConvertFrom-Json
                $progress += [pscustomobject]@{
                    shard = [int]$spec.shard
                    split = [string]$spec.split
                    completed_roots = [int]$heartbeat.completed_roots
                    target_roots = [int]$heartbeat.target_roots
                    updated_unix_seconds = $heartbeat.updated_unix_seconds
                }
            }
            catch { Write-Warning "Invalid heartbeat object: $uri" }
        }
    }
    catch { $progress = @() }

    $vmPrefix = ($RunName.ToLowerInvariant() -replace '[^a-z0-9-]', '-').Trim('-')
    if ($vmPrefix.Length -gt 54) { $vmPrefix = $vmPrefix.Substring(0, 54).Trim('-') }
    $runningInstances = @(gcloud compute instances list `
        --project $ProjectId `
        --filter ("name~'^" + $vmPrefix + "' AND status=RUNNING") `
        --format "value(name,zone,status)" 2>$null)
    $runningShards = @()
    foreach ($line in $runningInstances) {
        $name = ($line -split '\s+')[0]
        if ($name -match '-(\d{3})$') { $runningShards += [int]$matches[1] }
    }
    $runningShards = @($runningShards | Sort-Object -Unique)

    $missing = @()
    foreach ($spec in $specs) {
        if ($completedShards -notcontains [int]$spec.shard) { $missing += [int]$spec.shard }
    }
    $inactiveMissing = @($missing | Where-Object { $runningShards -notcontains $_ })
    $failedShards = @(
        $statusRecords |
            Where-Object { $_.status -eq "failed" -and $completedShards -notcontains [int]$_.shard } |
            ForEach-Object { [int]$_.shard } |
            Sort-Object -Unique
    )
    $doneRoots = [int](($doneRecords | Measure-Object roots -Sum).Sum)
    $partialRoots = [int](
        ($progress | Where-Object { $completedShards -notcontains [int]$_.shard } |
            Measure-Object completed_roots -Sum).Sum
    )
    $relaunch = if ($inactiveMissing.Count -gt 0) {
        ".\scripts\Start-GcpHuM42TeacherRun.ps1 -RunName '$RunName' -StartShards '$($inactiveMissing -join ',')' -CreateInstances"
    }
    else { $null }

    [pscustomobject]@{
        schema = "hu_m42_spot_status_report_v1"
        run_name = $RunName
        project_id = $ProjectId
        bucket = $Bucket
        expected_roots = [int]$manifest.total_roots
        completed_roots = $doneRoots
        resumable_partial_roots = $partialRoots
        expected_shards = $specs.Count
        completed_shards = $completedShards.Count
        completed_shard_indices = $completedShards
        running_shards = $runningShards
        failed_shards = $failedShards
        missing_count = $missing.Count
        missing_shards = @($missing | Select-Object -First $MaxMissingToShow)
        inactive_missing_shards = @($inactiveMissing | Select-Object -First $MaxMissingToShow)
        relaunch_command = $relaunch
        running_instances = $runningInstances
        resumable_progress = @($progress | Sort-Object shard)
        manifest_sha256 = $actualManifestSha256
        shards_manifest_sha256 = $actualShardsSha256
        manifest = $manifest
    } | ConvertTo-Json -Depth 10
}
finally {
    if (Test-Path $tmpRoot) { Remove-Item -LiteralPath $tmpRoot -Recurse -Force }
}
