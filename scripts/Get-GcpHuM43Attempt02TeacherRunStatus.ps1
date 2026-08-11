param(
    [string]$ProjectId = "ofc-solver-485418",
    [string]$Bucket = "pokerhu-ofc-solver-485418-training",
    [Parameter(Mandatory = $true)]
    [string]$RunName,
    [int]$MaxMissingToShow = 100
)

$ErrorActionPreference = "Stop"

if ($RunName -notmatch '^[a-zA-Z0-9][a-zA-Z0-9._-]*$') {
    throw "RunName may only contain letters, numbers, dot, underscore, and dash"
}
if ($MaxMissingToShow -lt 1) { throw "MaxMissingToShow must be positive" }

$cloudSdkGcloud = Join-Path $env:LOCALAPPDATA "Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
if (Test-Path -LiteralPath $cloudSdkGcloud) {
    Set-Alias -Name gcloud -Value $cloudSdkGcloud -Scope Script
}
elseif (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    throw "gcloud not found in PATH or at $cloudSdkGcloud"
}
$gcloudCommand = Get-Command gcloud -ErrorAction Stop
$gcloudExecutable = if ($gcloudCommand.CommandType -eq "Alias") {
    [string]$gcloudCommand.Definition
}
elseif ($gcloudCommand.Source) {
    [string]$gcloudCommand.Source
}
else {
    [string]$gcloudCommand.Name
}

function Get-Sha256 {
    param([string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-GcsUris {
    param([string]$Uri)
    $saved = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $output = @(& gcloud storage ls $Uri --project $ProjectId 2>&1)
        $code = $LASTEXITCODE
    }
    finally { $ErrorActionPreference = $saved }
    if ($code -eq 0) { return @($output | ForEach-Object { [string]$_ }) }
    $text = $output -join "`n"
    if ($text -match '(?i)(not found|no urls matched|matched no objects|404)') { return @() }
    throw "Unable to list GCS object(s) $Uri`: $text"
}

function Copy-GcsSnapshot {
    param(
        [object[]]$Items,
        [switch]$Required,
        [int]$MaxParallel = 15,
        [int]$TimeoutSeconds = 30,
        [int]$Attempts = 2
    )
    $work = @($Items)
    if ($work.Count -eq 0) { return @() }
    if ($MaxParallel -lt 1 -or $TimeoutSeconds -lt 1 -or $Attempts -lt 1) {
        throw "Invalid bounded GCS snapshot settings"
    }

    $workerCount = [Math]::Min($MaxParallel, $work.Count)
    $jobs = @()
    try {
        for ($worker = 0; $worker -lt $workerCount; $worker += 1) {
            $chunk = @()
            for ($index = $worker; $index -lt $work.Count; $index += $workerCount) {
                $chunk += $work[$index]
            }
            # Windows PowerShell's ConvertFrom-Json deliberately emits a JSON
            # array as one pipeline object.  Keep the wire shape explicitly an
            # array, then assign the decoded value before wrapping it below;
            # otherwise @($ItemsJson | ConvertFrom-Json) produces one nested
            # Object[] and casts all URI values into one space-joined string.
            $chunkJson = ConvertTo-Json -InputObject @($chunk) -Depth 5 -Compress
            $jobs += Start-Job -ScriptBlock {
                param($ItemsJson, $GcloudExecutable, $ProjectId, $Attempts)
                $decodedItems = ConvertFrom-Json -InputObject ([string]$ItemsJson)
                $items = @($decodedItems)
                foreach ($item in $items) {
                    $success = $false
                    $lastText = ""
                    $usedAttempts = 0
                    for ($attempt = 1; $attempt -le $Attempts; $attempt += 1) {
                        $usedAttempts = $attempt
                        $saved = $ErrorActionPreference
                        try {
                            $ErrorActionPreference = "Continue"
                            $output = @(& $GcloudExecutable storage cp ([string]$item.uri) `
                                ([string]$item.path) --project $ProjectId 2>&1)
                            $code = $LASTEXITCODE
                        }
                        finally { $ErrorActionPreference = $saved }
                        $lastText = $output -join "`n"
                        if ($code -eq 0) {
                            $success = $true
                            break
                        }
                        if ($attempt -lt $Attempts) { Start-Sleep -Milliseconds 250 }
                    }
                    [pscustomobject]@{
                        uri = [string]$item.uri
                        path = [string]$item.path
                        success = $success
                        attempts = $usedAttempts
                        error_text = $lastText
                    }
                }
            } -ArgumentList $chunkJson, $gcloudExecutable, $ProjectId, $Attempts
        }

        $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
        while (@($jobs | Where-Object { $_.State -in @("NotStarted", "Running") }).Count -gt 0 -and
            [DateTime]::UtcNow -lt $deadline) {
            Wait-Job -Job $jobs -Any -Timeout 1 | Out-Null
        }
        $unfinished = @($jobs | Where-Object { $_.State -in @("NotStarted", "Running") })
        foreach ($job in $unfinished) { Stop-Job -Job $job | Out-Null }
        $completedJobs = @($jobs | Where-Object { $_.State -eq "Completed" })
        $results = @($completedJobs | Receive-Job -ErrorAction SilentlyContinue)

        $reported = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
        foreach ($result in $results) { [void]$reported.Add([string]$result.uri) }
        foreach ($item in $work) {
            if (-not $reported.Contains([string]$item.uri)) {
                $results += [pscustomobject]@{
                    uri = [string]$item.uri
                    path = [string]$item.path
                    success = $false
                    attempts = 0
                    error_text = "bounded snapshot worker timed out or failed"
                }
            }
        }
        $failures = @($results | Where-Object { -not [bool]$_.success })
        if ($Required -and $failures.Count -gt 0) {
            $summary = @($failures | ForEach-Object { "$($_.uri): $($_.error_text)" }) -join "`n"
            throw "Required GCS snapshot failed: $summary"
        }
        return @($results)
    }
    finally {
        foreach ($job in $jobs) {
            if ($job.State -in @("NotStarted", "Running")) { Stop-Job -Job $job | Out-Null }
            Remove-Job -Job $job -Force -ErrorAction SilentlyContinue | Out-Null
        }
    }
}

function Assert-CloudBoundary {
    param($Manifest, [object[]]$Specs)
    $serialized = (@($Manifest) + @($Specs)) | ConvertTo-Json -Depth 20 -Compress
    if ($serialized -match '(?i)locked|attempt01|inherited[-_ ]holdout') {
        throw "Attempt02 cloud artifacts expose a sealed/local-only identifier"
    }
    $splitKeys = @($Manifest.split_roots.psobject.Properties.Name | Sort-Object)
    if (($splitKeys -join ',') -ne 'calibration,train') {
        throw "Attempt02 cloud manifest must expose only train/calibration splits"
    }
}

function Read-VerifiedDoneRecord {
    param(
        [string]$Path,
        [string]$Uri,
        $Spec,
        [string]$OutputPrefix,
        $Manifest,
        [string]$ManifestSha256,
        [string]$ShardsSha256
    )
    $done = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    if ($done.schema -ne "hu_m43_attempt02_teacher_done_v1" -or
        $done.status -ne "complete" -or
        [int]$done.shard -ne [int]$Spec.shard -or
        [string]$done.split -ne [string]$Spec.split -or
        [string]$done.output_prefix -ne $OutputPrefix -or
        [string]$done.manifest_sha256 -ne $ManifestSha256 -or
        [string]$done.shards_manifest_sha256 -ne $ShardsSha256 -or
        [string]$done.source_sha256 -ne [string]$Manifest.source_sha256 -or
        [string]$done.startup_sha256 -ne [string]$Manifest.startup_sha256 -or
        [string]$done.model_manifest_sha256 -ne [string]$Manifest.model_manifest_sha256 -or
        [string]$done.native_manifest_sha256 -ne [string]$Manifest.native_manifest_sha256) {
        throw "Invalid or stale Attempt02 DONE object: $Uri"
    }
    return $done
}

$prefix = "gs://$Bucket/runs/$RunName"
$tmpRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("hu-m43-a02-status-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $tmpRoot | Out-Null
try {
    $manifestPath = Join-Path $tmpRoot "manifest.json"
    $shardsPath = Join-Path $tmpRoot "shards_manifest.jsonl"
    [void]@(Copy-GcsSnapshot -Required -TimeoutSeconds 20 -Items @(
        [pscustomobject]@{ uri = "$prefix/manifest.json"; path = $manifestPath },
        [pscustomobject]@{ uri = "$prefix/source/shards_manifest.jsonl"; path = $shardsPath }
    ))

    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    if ($manifest.schema -ne "hu_m43_attempt02_teacher_spot_manifest_v1" -or
        [string]$manifest.run_name -ne $RunName -or
        [string]$manifest.project_id -ne $ProjectId -or
        [string]$manifest.bucket -ne $Bucket) {
        throw "Attempt02 manifest identity/schema mismatch"
    }
    $manifestSha256 = Get-Sha256 $manifestPath
    $shardsSha256 = Get-Sha256 $shardsPath
    if ([string]$manifest.shards_manifest_sha256 -ne $shardsSha256) {
        throw "Attempt02 shard manifest SHA256 mismatch"
    }

    $specs = @()
    foreach ($line in [System.IO.File]::ReadLines((Resolve-Path -LiteralPath $shardsPath))) {
        if ($line.Trim()) { $specs += ($line | ConvertFrom-Json) }
    }
    if ($specs.Count -ne [int]$manifest.total_shards) {
        throw "Attempt02 shard count disagrees with the frozen manifest"
    }
    $seenPrefixes = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    $seenSeeds = [Collections.Generic.HashSet[long]]::new()
    for ($index = 0; $index -lt $specs.Count; $index += 1) {
        $spec = $specs[$index]
        if ($spec.schema -ne "hu_m43_attempt02_teacher_shard_v1" -or
            [int]$spec.shard -ne $index -or
            [string]$spec.split -notin @("train", "calibration") -or
            -not $seenPrefixes.Add([string]$spec.output_prefix)) {
            throw "Attempt02 shard manifest identity/order is invalid at index $index"
        }
        for ($root = 0; $root -lt [int]$spec.roots; $root += 1) {
            $seed = [long]$spec.seed_start + ([long]$root * [long]$spec.seed_stride)
            if (-not $seenSeeds.Add($seed)) { throw "Duplicate fresh hand seed in cloud schedule: $seed" }
        }
    }
    Assert-CloudBoundary -Manifest $manifest -Specs $specs

    $specByPrefix = @{}
    foreach ($spec in $specs) { $specByPrefix[[string]$spec.output_prefix] = $spec }
    $doneRecords = @()
    $doneUris = @(Get-GcsUris "$prefix/results/*/DONE.json")
    $doneItems = @()
    foreach ($uri in $doneUris) {
        if ($uri -notmatch '/results/([^/]+)/DONE\.json$') { continue }
        $outputPrefix = $matches[1]
        if (-not $specByPrefix.ContainsKey($outputPrefix)) {
            throw "DONE URI does not map to one frozen shard: $uri"
        }
        $spec = $specByPrefix[$outputPrefix]
        $doneItems += [pscustomobject]@{
            uri = $uri
            path = Join-Path $tmpRoot ("done_{0:D3}.json" -f [int]$spec.shard)
            output_prefix = $outputPrefix
        }
    }
    # Thirty concurrent gcloud processes oversubscribe Windows PowerShell job
    # startup/authentication.  Six bounded workers keep exact-object copies
    # parallel while leaving enough time for five objects per worker.
    [void]@(Copy-GcsSnapshot -Items $doneItems -Required `
        -MaxParallel 6 -TimeoutSeconds 60)
    foreach ($item in $doneItems) {
        $uri = [string]$item.uri
        $outputPrefix = [string]$item.output_prefix
        $spec = $specByPrefix[$outputPrefix]
        $donePath = [string]$item.path
        $done = Read-VerifiedDoneRecord -Path $donePath -Uri $uri -Spec $spec `
            -OutputPrefix $outputPrefix -Manifest $manifest `
            -ManifestSha256 $manifestSha256 -ShardsSha256 $shardsSha256
        $doneRecords += $done
    }
    $completed = @($doneRecords | ForEach-Object { [int]$_.shard } | Sort-Object -Unique)

    # A single rsync provides a bounded bulk snapshot and preserves each shard
    # directory.  Checkpoints change less often than heartbeats and contain the
    # authoritative resumable root count; exclude the large partial JSONL and
    # rapidly replaced heartbeat objects.
    $resumeSnapshot = Join-Path $tmpRoot "resume_snapshot"
    New-Item -ItemType Directory -Force -Path $resumeSnapshot | Out-Null
    $saved = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $snapshotOutput = @(& gcloud storage rsync "$prefix/resume" $resumeSnapshot `
            --recursive --exclude '.*heartbeat[.]json$,.*teacher[.]jsonl[.]partial$' `
            --continue-on-error --project $ProjectId 2>&1)
        $snapshotCode = $LASTEXITCODE
    }
    finally { $ErrorActionPreference = $saved }
    if ($snapshotCode -ne 0) {
        $snapshotErrors = @($snapshotOutput | Where-Object { [string]$_ -match '^ERROR:' })
        $nonRaceErrors = @($snapshotErrors | Where-Object {
            [string]$_ -notmatch '(?i)(not found|404)'
        })
        if ($snapshotErrors.Count -eq 0 -or $nonRaceErrors.Count -gt 0) {
            throw "Unable to obtain bulk checkpoint snapshot: $($snapshotOutput -join '`n')"
        }
    }

    # Relist after rsync.  Any checkpoint that raced atomic replacement is
    # retried by exact URI in bounded parallel workers.  A second miss remains
    # explicitly unavailable rather than being reported as zero progress.
    $checkpointUris = @(Get-GcsUris "$prefix/resume/*/checkpoint.json")
    $checkpointRetryItems = @()
    foreach ($uri in $checkpointUris) {
        if ($uri -notmatch '/resume/([^/]+)/checkpoint\.json$') { continue }
        $outputPrefix = $matches[1]
        if (-not $specByPrefix.ContainsKey($outputPrefix)) { continue }
        $spec = $specByPrefix[$outputPrefix]
        if ($completed -contains [int]$spec.shard) { continue }
        $localPath = Join-Path (Join-Path $resumeSnapshot $outputPrefix) "checkpoint.json"
        if (-not (Test-Path -LiteralPath $localPath -PathType Leaf)) {
            New-Item -ItemType Directory -Force -Path (Split-Path -Parent $localPath) | Out-Null
            $checkpointRetryItems += [pscustomobject]@{
                uri = $uri
                path = $localPath
                output_prefix = $outputPrefix
            }
        }
    }
    [void]@(Copy-GcsSnapshot -Items $checkpointRetryItems -TimeoutSeconds 15)

    $progressByShard = @{}
    foreach ($uri in $checkpointUris) {
        if ($uri -notmatch '/resume/([^/]+)/checkpoint\.json$') { continue }
        $outputPrefix = $matches[1]
        if (-not $specByPrefix.ContainsKey($outputPrefix)) { continue }
        $spec = $specByPrefix[$outputPrefix]
        if ($completed -contains [int]$spec.shard) { continue }
        $checkpointPath = Join-Path (Join-Path $resumeSnapshot $outputPrefix) "checkpoint.json"
        if (-not (Test-Path -LiteralPath $checkpointPath -PathType Leaf)) {
            continue
        }
        try {
            $checkpoint = Get-Content -LiteralPath $checkpointPath -Raw | ConvertFrom-Json
            $completedRoots = [int]$checkpoint.completed_roots
            if ($checkpoint.schema -ne "hu_m4_t1_second_checkpoint_v1" -or
                [int]$checkpoint.target_roots -ne [int]$spec.roots -or
                $completedRoots -lt 0 -or $completedRoots -gt [int]$spec.roots -or
                [string]$checkpoint.config_sha256 -notmatch '^[0-9a-f]{64}$' -or
                [string]$checkpoint.partial_sha256 -notmatch '^[0-9a-f]{64}$') {
                throw "checkpoint identity/progress mismatch"
            }
            $progressByShard[[int]$spec.shard] = [pscustomobject]@{
                shard = [int]$spec.shard
                split = [string]$spec.split
                completed_roots = $completedRoots
                target_roots = [int]$spec.roots
                updated_unix_seconds = $checkpoint.updated_unix_seconds
                source = "checkpoint"
            }
        }
        catch {
            Write-Warning "Invalid checkpoint object: $uri"
        }
    }
    $progress = @($progressByShard.Values)

    $vmPrefix = ($RunName.ToLowerInvariant() -replace '[^a-z0-9-]', '-').Trim('-')
    if ($vmPrefix.Length -gt 54) { $vmPrefix = $vmPrefix.Substring(0, 54).Trim('-') }
    $saved = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $runningInstances = @(& gcloud compute instances list --project $ProjectId `
            --filter ("name~'^" + $vmPrefix + "-' AND status=RUNNING") `
            --format "value(name,zone,status)" 2>&1)
        $runningCode = $LASTEXITCODE
    }
    finally { $ErrorActionPreference = $saved }
    if ($runningCode -ne 0) {
        throw "Unable to obtain authoritative RUNNING VM snapshot: $($runningInstances -join '`n')"
    }
    $running = @()
    foreach ($line in $runningInstances) {
        $name = ([string]$line -split '\s+')[0]
        if ($name -match '-(\d{3})$') { $running += [int]$matches[1] }
    }
    $running = @($running | Sort-Object -Unique)

    # A worker uploads immutable DONE and then self-deletes.  It can therefore
    # be absent from the initial DONE snapshot, appear STOPPING in the VM
    # snapshot, and otherwise be misclassified as inactive.  Recheck only
    # those provisional inactive shards before producing a relaunch command.
    $preliminaryMissing = @(0..($specs.Count - 1) | Where-Object {
        $completed -notcontains $_
    })
    $preliminaryInactive = @($preliminaryMissing | Where-Object {
        $running -notcontains $_
    })
    $lateDoneItems = @()
    foreach ($shard in $preliminaryInactive) {
        $spec = $specs[[int]$shard]
        $lateDoneItems += [pscustomobject]@{
            uri = "$prefix/results/$([string]$spec.output_prefix)/DONE.json"
            path = Join-Path $tmpRoot ("late_done_{0:D3}.json" -f [int]$shard)
            output_prefix = [string]$spec.output_prefix
            shard = [int]$shard
        }
    }
    $lateDoneItemByUri = @{}
    foreach ($item in $lateDoneItems) { $lateDoneItemByUri[[string]$item.uri] = $item }
    $lateDoneReclassified = @()
    foreach ($copy in @(Copy-GcsSnapshot -Items $lateDoneItems `
        -TimeoutSeconds 20 -Attempts 2)) {
        if (-not [bool]$copy.success) { continue }
        $item = $lateDoneItemByUri[[string]$copy.uri]
        $spec = $specs[[int]$item.shard]
        $done = Read-VerifiedDoneRecord -Path ([string]$item.path) `
            -Uri ([string]$item.uri) -Spec $spec `
            -OutputPrefix ([string]$item.output_prefix) -Manifest $manifest `
            -ManifestSha256 $manifestSha256 -ShardsSha256 $shardsSha256
        $doneRecords += $done
        $lateDoneReclassified += [int]$item.shard
    }
    $lateDoneReclassified = @($lateDoneReclassified | Sort-Object -Unique)
    $completed = @($doneRecords | ForEach-Object { [int]$_.shard } | Sort-Object -Unique)
    $running = @($running | Where-Object { $completed -notcontains $_ })
    $missing = @(0..($specs.Count - 1) | Where-Object { $completed -notcontains $_ })
    $inactive = @($missing | Where-Object { $running -notcontains $_ })
    $progress = @($progress | Where-Object { $completed -notcontains [int]$_.shard })
    $progressUnavailableShards = @($missing | Where-Object {
        -not $progressByShard.ContainsKey([int]$_)
    })
    $relaunch = if ($inactive.Count -gt 0) {
        ".\scripts\Start-GcpHuM43Attempt02TeacherRun.ps1 -RunName '$RunName' -StartShards '$($inactive -join ',')' -CreateInstances"
    }
    else { $null }

    [pscustomobject]@{
        schema = "hu_m43_attempt02_teacher_spot_status_v1"
        run_name = $RunName
        expected_roots = [int]$manifest.total_roots
        completed_roots = [int](($doneRecords | Measure-Object roots -Sum).Sum)
        resumable_partial_roots = [int](($progress | Measure-Object completed_roots -Sum).Sum)
        expected_shards = $specs.Count
        completed_shards = $completed.Count
        completed_shard_indices = $completed
        late_done_reclassified_shards = $lateDoneReclassified
        running_shards = $running
        missing_count = $missing.Count
        missing_shards = @($missing | Select-Object -First $MaxMissingToShow)
        inactive_missing_shards = @($inactive | Select-Object -First $MaxMissingToShow)
        relaunch_command = $relaunch
        resumable_progress = @($progress | Sort-Object shard)
        progress_snapshot_complete = ($progressUnavailableShards.Count -eq 0)
        progress_snapshot_available_shards = $progress.Count
        progress_snapshot_unavailable_shards = @($progressUnavailableShards | Select-Object -First $MaxMissingToShow)
        manifest_sha256 = $manifestSha256
        shards_manifest_sha256 = $shardsSha256
    } | ConvertTo-Json -Depth 10
}
finally {
    if (Test-Path -LiteralPath $tmpRoot) { Remove-Item -LiteralPath $tmpRoot -Recurse -Force }
}
