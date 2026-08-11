param(
    [Parameter(Mandatory = $true)][string]$RunName,
    [string]$ProjectId = "ofc-solver-485418",
    [string]$Bucket = "pokerhu-ofc-solver-485418-training"
)

$ErrorActionPreference = "Stop"

if ($RunName -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$') {
    throw "RunName may only contain letters, numbers, dot, underscore, and dash"
}

function Get-Sha256 {
    param([string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-GcsJson {
    param([string]$Uri)
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $raw = & gcloud storage cat $Uri 2>&1
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($exitCode -eq 0) {
        return (@($raw) -join "`n") | ConvertFrom-Json
    }
    $message = @($raw) -join [Environment]::NewLine
    if ($message -match '(?i)not found|does not exist|No URLs matched|matched no objects|404') {
        return $null
    }
    throw "Unable to read GCS status object: $Uri`n$message"
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runDir = Join-Path $repoRoot "outputs/gcp_runs/$RunName"
$manifestPath = Join-Path $runDir "model_run_manifest.json"
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "Local frozen M4.2 model manifest is missing: $manifestPath"
}
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$manifestSha = Get-Sha256 $manifestPath
if ($manifest.schema -ne "hu_m42_model_run_manifest_v1" -or
    $manifest.run_name -ne $RunName -or
    $manifest.project_id -ne $ProjectId -or
    $manifest.bucket -ne $Bucket -or
    $manifest.no_runtime_activation -ne $true) {
    throw "Local M4.2 model manifest identity or activation boundary is invalid"
}
$vmName = if ($null -ne $manifest.compute.vm_name -and [string]$manifest.compute.vm_name) {
    [string]$manifest.compute.vm_name
}
else {
    $normalized = ($RunName.ToLowerInvariant() -replace '[^a-z0-9-]', '-').Trim('-')
    if ($normalized.Length -gt 63) { $normalized = $normalized.Substring(0, 63).TrimEnd('-') }
    $normalized
}

$prefix = "gs://$Bucket/runs/$RunName/results"
$done = Get-GcsJson "$prefix/DONE"
$status = Get-GcsJson "$prefix/status.json"
$heartbeat = Get-GcsJson "$prefix/heartbeat.json"
if ($null -ne $done) {
    if ($done.schema -ne "hu_m42_model_done_v1" -or
        $done.status -ne "complete" -or
        $done.run_name -ne $RunName -or
        $done.run_manifest_sha256 -ne $manifestSha -or
        $done.no_runtime_activation -ne $true) {
        throw "Invalid or stale M4.2 model DONE object"
    }
}
if ($null -ne $status -and
    ($status.schema -ne "hu_m42_model_status_v1" -or
        $status.run_name -ne $RunName -or
        ($null -ne $status.run_manifest_sha256 -and $status.run_manifest_sha256 -ne $manifestSha))) {
    throw "Invalid or stale M4.2 model status object"
}
if ($null -ne $heartbeat -and
    ($heartbeat.schema -ne "hu_m42_model_heartbeat_v1" -or
        $heartbeat.run_name -ne $RunName -or
        ($null -ne $heartbeat.run_manifest_sha256 -and $heartbeat.run_manifest_sha256 -ne $manifestSha))) {
    throw "Invalid or stale M4.2 model heartbeat object"
}

$previousPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    $instanceRaw = & gcloud compute instances list --project $ProjectId --filter ("name={0}" -f $vmName) --format=json 2>&1
    $instanceExit = $LASTEXITCODE
}
finally {
    $ErrorActionPreference = $previousPreference
}
if ($instanceExit -ne 0) {
    throw "Unable to list M4.2 model worker: $(@($instanceRaw) -join [Environment]::NewLine)"
}
$instances = (@($instanceRaw) -join "`n") | ConvertFrom-Json

[ordered]@{
    schema = "hu_m42_model_status_report_v1"
    run_name = $RunName
    state = $(
        if ($null -ne $done) { "complete" }
        elseif ($null -ne $status -and [string]$status.state -eq "complete") { "finalizing_without_done" }
        elseif ($null -ne $status) { [string]$status.state }
        else { "pending" }
    )
    manifest_sha256 = $manifestSha
    done_commit_present = ($null -ne $done)
    done = $done
    status = $status
    heartbeat = $heartbeat
    active_instances = @($instances).Count
    instances = @($instances | ForEach-Object {
        [ordered]@{
            name = $_.name
            zone = ([string]$_.zone).Split('/')[-1]
            status = $_.status
            machine_type = ([string]$_.machineType).Split('/')[-1]
        }
    })
    no_runtime_activation = $true
} | ConvertTo-Json -Depth 12
