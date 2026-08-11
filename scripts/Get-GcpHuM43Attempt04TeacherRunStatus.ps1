param(
    [Parameter(Mandatory = $true)][string]$RunName,
    [Parameter(Mandatory = $true)]
    [ValidateSet(
        'precal_holdout',
        'calibration_safety_fit',
        'calibration_threshold_lock',
        'locked_holdout'
    )][string]$Role,
    [string]$ProjectId = 'ofc-solver-485418',
    [string]$Bucket = 'pokerhu-ofc-solver-485418-training',
    [string]$RunDir
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'HuM43Attempt04Spot.Common.ps1')

if ($RunName -notmatch '^[a-z0-9][a-z0-9-]{2,126}[a-z0-9]$') {
    throw 'RunName is not a safe GCP run identity'
}

function Assert-M43A4TeacherDone {
    param(
        [Parameter(Mandatory = $true)]$Done,
        [Parameter(Mandatory = $true)]$Spec,
        [Parameter(Mandatory = $true)]$Manifest,
        [Parameter(Mandatory = $true)][string]$ManifestSha256
    )
    if ($Done.schema -ne 'hu_m43_attempt02_teacher_done_v1' -or
        $Done.status -ne 'complete' -or
        [string]$Done.run_name -ne $RunName -or
        [int]$Done.shard -ne [int]$Spec.shard -or
        [string]$Done.split -ne [string]$Spec.split -or
        [int]$Done.roots -ne [int]$Spec.roots -or
        [string]$Done.output_prefix -ne [string]$Spec.output_prefix -or
        [string]$Done.manifest_sha256 -ne $ManifestSha256 -or
        [string]$Done.shards_manifest_sha256 -ne [string]$Manifest.schedule_sha256 -or
        [string]$Done.source_sha256 -ne [string]$Manifest.source_sha256 -or
        [string]$Done.startup_sha256 -ne [string]$Manifest.startup_sha256 -or
        [string]$Done.model_manifest_sha256 -ne [string]$Manifest.model_manifest_sha256 -or
        [string]$Done.native_manifest_sha256 -ne [string]$Manifest.native_manifest_sha256) {
        throw "Attempt04 teacher DONE does not match shard $($Spec.shard)"
    }
    foreach ($value in @(
        $Done.output_sha256,
        $Done.checkpoint_sha256,
        $Done.heartbeat_sha256
    )) {
        Assert-M43A4Sha256 $value 'teacher DONE artifact hash'
    }
}

function Read-M43A4RemoteJsonExact {
    param([Parameter(Mandatory = $true)][string]$Uri)
    $temporary = Join-Path (
        [IO.Path]::GetTempPath()
    ) ('m43a4-status-' + [guid]::NewGuid().ToString('N') + '.json')
    try {
        Copy-M43A4RemoteFileExact `
            -Uri $Uri -Destination $temporary -ProjectId $ProjectId
        return (Get-Content -LiteralPath $temporary -Raw | ConvertFrom-Json)
    }
    finally {
        Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $RunDir) {
    $RunDir = Join-Path (Join-Path $repoRoot 'outputs/gcp_runs') $RunName
}
$RunDir = Resolve-M43A4Path -Path $RunDir -Root $repoRoot -Label 'Attempt04 run directory'
Assert-M43A4UnderRoot -Path $RunDir -Root $repoRoot -Label 'Attempt04 run directory'
$manifestPath = Join-Path $RunDir 'manifest.json'
$schedulePath = Join-Path $RunDir 'shards_manifest.jsonl'
foreach ($path in @($manifestPath, $schedulePath)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Attempt04 immutable run input is missing: $path"
    }
}
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($manifest.schema -ne 'hu_m43_attempt04_teacher_spot_manifest_v1' -or
    $manifest.status -ne 'frozen' -or
    [string]$manifest.run_name -ne $RunName -or
    [string]$manifest.project_id -ne $ProjectId -or
    [string]$manifest.bucket -ne $Bucket -or
    [int]$manifest.total_shards -ne 70 -or
    [int]$manifest.total_roots -ne 700 -or
    [string]$manifest.schedule_sha256 -ne (Get-M43A4Sha256 $schedulePath)) {
    throw 'Attempt04 local run closure changed'
}
$manifestSha256 = Get-M43A4Sha256 $manifestPath
$prefix = "gs://$Bucket/runs/$RunName"
$temporaryRoot = Join-Path (
    [IO.Path]::GetTempPath()
) ('m43a4-status-closure-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $temporaryRoot | Out-Null
try {
    Copy-M43A4RemoteFileExact `
        -Uri "$prefix/manifest.json" `
        -Destination (Join-Path $temporaryRoot 'manifest.json') `
        -ProjectId $ProjectId `
        -ExpectedSha256 $manifestSha256
    Copy-M43A4RemoteFileExact `
        -Uri "$prefix/source/shards_manifest.jsonl" `
        -Destination (Join-Path $temporaryRoot 'shards_manifest.jsonl') `
        -ProjectId $ProjectId `
        -ExpectedSha256 ([string]$manifest.schedule_sha256)
}
finally {
    Remove-Item -LiteralPath $temporaryRoot -Recurse -Force -ErrorAction SilentlyContinue
}

$allSpecs = @(
    [IO.File]::ReadLines($schedulePath) |
        Where-Object { $_.Trim() } |
        ForEach-Object { $_ | ConvertFrom-Json }
)
if ($allSpecs.Count -ne 70) { throw 'Attempt04 schedule is not exact 70 shards' }
$roleSpecs = @($allSpecs | Where-Object { [string]$_.logical_role -eq $Role })
$expectedRoleShards = [ordered]@{
    precal_holdout = 30
    calibration_safety_fit = 10
    calibration_threshold_lock = 10
    locked_holdout = 20
}
if ($roleSpecs.Count -ne [int]$expectedRoleShards[$Role] -or
    @($roleSpecs | Where-Object { [string]$_.split -ne 'train' }).Count -ne 0) {
    throw "Attempt04 schedule role mapping changed: $Role"
}

# Result URIs are formed only after the schedule has been reduced to one role.
$rows = foreach ($spec in $roleSpecs) {
    $doneUri = "$prefix/results/$($spec.output_prefix)/DONE.json"
    $present = Test-M43A4GcsObject -Uri $doneUri -ProjectId $ProjectId
    if ($present) {
        $done = Read-M43A4RemoteJsonExact -Uri $doneUri
        Assert-M43A4TeacherDone `
            -Done $done -Spec $spec -Manifest $manifest `
            -ManifestSha256 $manifestSha256
    }
    [pscustomobject][ordered]@{
        shard = [int]$spec.shard
        role_shard = [int]$spec.role_shard
        logical_role = [string]$spec.logical_role
        output_prefix = [string]$spec.output_prefix
        state = $(if ($present) { 'complete_verified_done' } else { 'not_complete' })
    }
}
$complete = @($rows | Where-Object { $_.state -eq 'complete_verified_done' }).Count
[pscustomobject][ordered]@{
    schema = 'hu_m43_attempt04_teacher_role_status_v1'
    run_name = $RunName
    logical_role = $Role
    expected_shards = $roleSpecs.Count
    complete_shards = $complete
    incomplete_shards = $roleSpecs.Count - $complete
    all_done_markers_verified = ($complete -eq $roleSpecs.Count)
    manifest_sha256 = $manifestSha256
    schedule_sha256 = [string]$manifest.schedule_sha256
    teacher_payload_downloaded = $false
    other_role_result_objects_addressed = $false
    shards = $rows
    current_profile_mutated = $false
    runtime_policy_activated = $false
} | ConvertTo-Json -Depth 8
