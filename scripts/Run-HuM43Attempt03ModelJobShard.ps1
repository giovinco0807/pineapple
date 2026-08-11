param(
    [Parameter(Mandatory = $true)][string]$Schedule,
    [Parameter(Mandatory = $true)][ValidateRange(0, 7)][int]$ShardIndex,
    [string]$ReceiptOutput
)

$ErrorActionPreference = "Stop"

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Write-Utf8CreateNew {
    param([Parameter(Mandatory = $true)][string]$Path, [Parameter(Mandatory = $true)][string]$Text)
    $parent = Split-Path -Parent $Path
    if ($parent) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
    $bytes = [Text.UTF8Encoding]::new($false).GetBytes($Text)
    $stream = [IO.File]::Open($Path, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
    try { $stream.Write($bytes, 0, $bytes.Length); $stream.Flush($true) }
    finally { $stream.Dispose() }
}

$schedulePath = (Resolve-Path -LiteralPath $Schedule).Path
$payload = Get-Content -LiteralPath $schedulePath -Raw | ConvertFrom-Json
if ($payload.schema -ne "hu_m43_attempt03_v5_model_job_schedule_v1" -or
    $payload.status -ne "frozen_commands_not_started" -or
    [int]$payload.job_count -ne 30 -or [int]$payload.shard_count -ne 8 -or
    [int]$payload.jobs_per_shard -ne 4 -or [int]$payload.canary_shard -ne 0 -or
    [bool]$payload.fanout_before_canary_done_allowed -or
    [bool]$payload.current_profile_mutated -or [bool]$payload.runtime_policy_activated) {
    throw "Attempt03 model job schedule lifecycle changed"
}
if ([string]$payload.model_freeze_file_sha256 -notmatch '^[0-9a-f]{64}$' -or
    [string]$payload.training_freeze_file_sha256 -notmatch '^[0-9a-f]{64}$') {
    throw "Attempt03 model job schedule freeze lineage changed"
}
$allJobs = @($payload.jobs | Sort-Object job_index)
if ($allJobs.Count -ne 30 -or (($allJobs.job_index -join ',') -ne ((0..29) -join ','))) {
    throw "Attempt03 model schedule lost exact 30-job coverage"
}
$jobs = @($allJobs | Where-Object { [int]$_.shard_index -eq $ShardIndex })
$expectedCount = if ($ShardIndex -eq 7) { 2 } else { 4 }
if ($jobs.Count -ne $expectedCount) { throw "Attempt03 shard job partition changed" }

foreach ($property in $payload.process_environment.PSObject.Properties) {
    [Environment]::SetEnvironmentVariable([string]$property.Name, [string]$property.Value, "Process")
}

$completed = @()
foreach ($job in $jobs) {
    $argv = @($job.argv | ForEach-Object { [string]$_ })
    if ($argv.Count -lt 5 -or $argv[0] -notin @("python", "python3") -or
        $argv[1] -ne "-B" -or $argv[2] -ne "-m" -or
        $argv[3] -ne "ofc_regular.train_hu_m43_attempt03_fold_job") {
        throw "Attempt03 worker argv boundary changed for job $($job.job_index)"
    }
    & $argv[0] @($argv | Select-Object -Skip 1)
    if ($LASTEXITCODE -ne 0) { throw "Attempt03 model job failed: $($job.job_index)" }
    $done = Join-Path ([string]$job.output_dir) "DONE.json"
    if (-not (Test-Path -LiteralPath $done -PathType Leaf)) {
        throw "Attempt03 model job did not publish DONE: $($job.job_index)"
    }
    $donePayload = Get-Content -LiteralPath $done -Raw | ConvertFrom-Json
    if ($donePayload.schema -ne "hu_m43_attempt03_v5_fold_done_v1" -or
        $donePayload.status -ne "complete" -or
        [int]$donePayload.job_index -ne [int]$job.job_index -or
        [string]$donePayload.job_spec_sha256 -ne [string]$job.job_spec_sha256 -or
        [bool]$donePayload.current_profile_mutated -or
        [bool]$donePayload.runtime_policy_activated) {
        throw "Attempt03 model job DONE binding changed: $($job.job_index)"
    }
    $completed += [ordered]@{
        job_index = [int]$job.job_index
        job_spec_sha256 = [string]$job.job_spec_sha256
        done_path = $done
        done_file_sha256 = Get-Sha256 $done
    }
}

$receipt = [ordered]@{
    schema = "hu_m43_attempt03_v5_model_job_shard_receipt_v1"
    status = "complete"
    run_name = [string]$payload.run_name
    shard_index = $ShardIndex
    canary = ($ShardIndex -eq 0)
    schedule_file_sha256 = Get-Sha256 $schedulePath
    fold_cloud_contract_sha256 = [string]$payload.fold_cloud_contract_sha256
    training_config_sha256 = [string]$payload.training_config_sha256
    model_freeze_file_sha256 = [string]$payload.model_freeze_file_sha256
    training_freeze_file_sha256 = [string]$payload.training_freeze_file_sha256
    jobs = $completed
    current_profile_mutated = $false
    runtime_policy_activated = $false
}
if ($ReceiptOutput) {
    $receiptPath = [IO.Path]::GetFullPath($ReceiptOutput)
    Write-Utf8CreateNew -Path $receiptPath -Text (($receipt | ConvertTo-Json -Depth 12) + "`n")
}
$receipt | ConvertTo-Json -Depth 12
