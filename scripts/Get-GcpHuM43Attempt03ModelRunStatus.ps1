param(
    [string]$ProjectId = "ofc-solver-485418",
    [string]$Bucket = "pokerhu-ofc-solver-485418-training",
    [Parameter(Mandatory = $true)][string]$RunName,
    [string]$Manifest,
    [switch]$IncludeInstances
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "HuM43Attempt03ModelSpot.Common.ps1")

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ($RunName -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]{2,119}$') { throw "RunName is unsafe" }
$prefix = "gs://$Bucket/runs/$RunName"
$manifestUri = "$prefix/source/m43_attempt03_v5_model_spot_manifest.json"
$temporary = Join-Path ([IO.Path]::GetTempPath()) ("m43-a3-status-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $temporary | Out-Null
try {
    $manifestPath = Join-Path $temporary "manifest.json"
    if ($Manifest) {
        $resolved = Resolve-M43A3Input $Manifest $repoRoot "Attempt03 model Spot manifest"
        [IO.File]::WriteAllBytes($manifestPath, [IO.File]::ReadAllBytes($resolved))
    }
    else {
        Invoke-M43A3Gcloud @("storage", "cp", $manifestUri, $manifestPath, "--project", $ProjectId) 60 | Out-Null
    }
    $manifestSha = Get-M43A3Sha256 $manifestPath
    $run = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    Assert-M43A3RunManifest $run $RunName $ProjectId $Bucket

    $completed = @(); $jobStates = @(); $seenDone = [Collections.Generic.HashSet[int]]::new()
    for ($index = 0; $index -lt 30; $index++) {
        $job = @($run.jobs)[$index]
        if (Test-M43A3GcsObject $job.done_uri) {
            $done = Get-M43A3GcsJson $job.done_uri
            Assert-M43A3Done $done $run $index $manifestSha
            if (-not $seenDone.Add($index)) { throw "Duplicate DONE job identity: $index" }
            $completed += $index
            $jobStates += [ordered]@{ job_index = $index; shard_index = [int]$job.shard_index; state = "complete"; done_hash_chain = "pass" }
            continue
        }
        $statusUri = "$prefix/results/job-$($index.ToString('D2'))/status.json"
        $heartbeatUri = "$prefix/results/job-$($index.ToString('D2'))/heartbeat.json"
        $state = "pending"
        $updatedAt = $null
        if (Test-M43A3GcsObject $statusUri) {
            $status = Get-M43A3GcsJson $statusUri
            if ($status.schema -ne "hu_m43_attempt03_v5_model_spot_job_status_v1" -or
                [string]$status.run_name -ne $RunName -or
                (Get-M43A3StrictInteger $status.job_index "status.job_index") -ne $index -or
                [string]$status.run_manifest_sha256 -ne $manifestSha -or
                [string]$status.source_sha256 -ne [string]$run.source.sha256 -or
                $status.current_profile_mutated -ne $false -or $status.runtime_policy_activated -ne $false -or
                [string]$status.state -notin @("running", "failed", "complete")) {
                throw "Attempt03 job status lineage changed: $index"
            }
            $state = [string]$status.state; $updatedAt = [string]$status.updated_at
            if ($state -eq "complete") { $state = "complete_status_without_done" }
        }
        elseif (Test-M43A3GcsObject $heartbeatUri) {
            $heartbeat = Get-M43A3GcsJson $heartbeatUri
            if ($heartbeat.schema -ne "hu_m43_attempt03_v5_model_spot_job_heartbeat_v1" -or
                [string]$heartbeat.run_name -ne $RunName -or
                (Get-M43A3StrictInteger $heartbeat.job_index "heartbeat.job_index") -ne $index -or
                [string]$heartbeat.run_manifest_sha256 -ne $manifestSha) {
                throw "Attempt03 job heartbeat lineage changed: $index"
            }
            $state = "running"; $updatedAt = [string]$heartbeat.updated_at
        }
        $jobStates += [ordered]@{ job_index = $index; shard_index = [int]$job.shard_index; state = $state; updated_at = $updatedAt }
    }

    $shardStates = @(); $completedShards = @()
    for ($shard = 0; $shard -lt 8; $shard++) {
        $expected = @($run.jobs | Where-Object { [int]$_.shard_index -eq $shard } | ForEach-Object { [int]$_.job_index })
        $receiptUri = "$prefix/results/shard-$($shard.ToString('D2'))/receipt.json"
        $receiptPresent = Test-M43A3GcsObject $receiptUri
        $jobsComplete = @($expected | Where-Object { $_ -in $completed }).Count -eq $expected.Count
        if ($receiptPresent) {
            $receipt = Get-M43A3GcsJson $receiptUri
            if ($receipt.schema -ne "hu_m43_attempt03_v5_model_spot_shard_receipt_v1" -or
                $receipt.status -ne "complete" -or [string]$receipt.run_name -ne $RunName -or
                (Get-M43A3StrictInteger $receipt.shard_index "receipt.shard_index") -ne $shard -or
                [string]$receipt.run_manifest_sha256 -ne $manifestSha -or
                [string]$receipt.source_sha256 -ne [string]$run.source.sha256 -or
                (@($receipt.jobs) -join ",") -ne ($expected -join ",") -or
                $receipt.current_profile_mutated -ne $false -or $receipt.runtime_policy_activated -ne $false) {
                throw "Attempt03 shard receipt lineage changed: $shard"
            }
            if (-not $jobsComplete) { throw "Attempt03 shard receipt exists before all strict DONE artifacts: $shard" }
            $completedShards += $shard
        }
        $shardStates += [ordered]@{
            shard_index = $shard; expected_jobs = $expected; jobs_complete = $jobsComplete
            receipt_present = $receiptPresent; state = if ($receiptPresent -and $jobsComplete) { "complete" } elseif ($jobsComplete) { "done_without_receipt" } else { "incomplete" }
        }
    }

    $instances = @()
    if ($IncludeInstances) {
        $raw = Invoke-M43A3Gcloud @("compute", "instances", "list", "--project", $ProjectId, "--filter", "name~'$($run.compute.vm_prefix)-s'", "--format=json") 30
        if ($raw) { $instances = @(((@($raw) -join "`n") | ConvertFrom-Json)) }
    }
    $allComplete = $completed.Count -eq 30 -and $completedShards.Count -eq 8
    $active = @($instances | Where-Object { [string]$_.status -in @("PROVISIONING", "STAGING", "RUNNING", "STOPPING") }).Count
    $overall = if ($allComplete) { "complete" } elseif ($active -gt 0) { "running" } else { "incomplete_inactive" }
    [ordered]@{
        schema = "hu_m43_attempt03_v5_model_spot_status_v1"; status = $overall
        run_name = $RunName; manifest_sha256 = $manifestSha
        completed_jobs = $completed; completed_job_count = $completed.Count
        missing_jobs = @(0..29 | Where-Object { $_ -notin $completed })
        completed_shards = $completedShards; canary_complete = 0 -in $completedShards
        fanout_allowed = 0 -in $completedShards; job_states = $jobStates; shard_states = $shardStates
        active_instance_count = $active; instances = $instances
        strict_done_and_receipt_validation = "pass"; current_profile_mutated = $false
        runtime_policy_activated = $false; full_replacement = $false
    } | ConvertTo-Json -Depth 20
}
finally { Remove-Item -LiteralPath $temporary -Recurse -Force -ErrorAction SilentlyContinue }
