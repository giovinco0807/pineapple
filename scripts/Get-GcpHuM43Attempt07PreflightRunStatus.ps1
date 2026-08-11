param(
    [Parameter(Mandatory = $true)][string]$RunName,
    [string[]]$Jobs = @('0-4'),
    [string]$ProjectId = 'ofc-solver-485418',
    [string]$Bucket = 'pokerhu-ofc-solver-485418-training',
    [string]$RunDir
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'HuM43Attempt04Spot.Common.ps1')

$ExpectedAttempt07PlanSha256 = '8bf15f8d109a2e441e1c240e533c56df42a7e65d2eefd3b2750d1bd63758b189'
$ExpectedSourceSha256 = '6b1063589aa2ee4f3e65d9489384176a85c69abe1dfefe30ffb4474f96964903'
$ExpectedModelSha256 = 'e7fa728308c8973e2e202baf79309e30b9b6f58c37431d8ea55850390abc28c3'
$ExpectedAiProfilesSha256 = 'd2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3'
$JobCount = 5
$ExpectedDoneFields = @(
    'schema', 'status', 'run_name', 'job_index', 'job_id', 'source_root_index',
    'batch_child_selectors', 'native_batch_threads', 'output_prefix',
    'output_sha256', 'checkpoint_sha256', 'heartbeat_sha256', 'summary_sha256',
    'run_log_sha256', 'manifest_sha256', 'authorization_sha256', 'schedule_sha256',
    'attempt07_plan_sha256', 'source_merged_sha256', 'model_sha256',
    'ai_profiles_sha256', 'elapsed_seconds', 'peak_rss_bytes',
    'teacher_values_exported', 'arm_selection_performed', 'new_root_generated',
    'current_profile_resolved', 'current_profile_mutated', 'runtime_policy_activated'
) | Sort-Object

function Expand-M43A7StatusSelection {
    param([string[]]$Values)
    $selected = [Collections.Generic.SortedSet[int]]::new()
    foreach ($raw in $Values) {
        foreach ($token in ([string]$raw -split ',')) {
            $value = $token.Trim()
            if (-not $value) { continue }
            if ($value -match '^(\d+)-(\d+)$') {
                $first = [int]$Matches[1]
                $last = [int]$Matches[2]
                if ($first -gt $last) { throw "Invalid Attempt07 job range: $value" }
                for ($index = $first; $index -le $last; $index++) { [void]$selected.Add($index) }
            }
            elseif ($value -match '^\d+$') { [void]$selected.Add([int]$value) }
            else { throw "Invalid Attempt07 job selection: $value" }
        }
    }
    foreach ($index in $selected) {
        if ($index -lt 0 -or $index -ge $JobCount) { throw "Attempt07 job outside 0..4: $index" }
    }
    return @($selected)
}

function Test-M43A7Integer {
    param($Value)
    return ($Value -is [byte] -or $Value -is [sbyte] -or
        $Value -is [int16] -or $Value -is [uint16] -or
        $Value -is [int32] -or $Value -is [uint32] -or
        $Value -is [int64] -or $Value -is [uint64])
}

function Assert-M43A7False {
    param($Value, [string]$Label)
    if ($Value -isnot [bool] -or $Value -ne $false) { throw "$Label must be boolean false" }
}

function Read-M43A7DoneExact {
    param([string]$Uri)
    $temporary = Join-Path ([IO.Path]::GetTempPath()) ('m43a7-status-done-' + [guid]::NewGuid().ToString('N') + '.json')
    try {
        Copy-M43A4RemoteFileExact -Uri $Uri -Destination $temporary -ProjectId $ProjectId
        return (Get-Content -LiteralPath $temporary -Raw | ConvertFrom-Json)
    }
    finally { Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue }
}

function Assert-M43A7Authorization {
    param($Authorization, [string]$ManifestSha256, $Manifest)
    if ($Authorization.schema -ne 'hu_m43_attempt07_preflight_spot_launch_authorization_v1' -or
        $Authorization.status -ne 'authorized_for_bounded_spot_preflight' -or
        [string]$Authorization.run_name -ne $RunName -or
        [string]$Authorization.manifest_sha256 -ne $ManifestSha256 -or
        [string]$Authorization.preflight_plan_sha256 -ne [string]$Manifest.preflight_plan_sha256 -or
        [string]$Authorization.attempt07_plan_sha256 -ne $ExpectedAttempt07PlanSha256 -or
        [string]$Authorization.source_merged_sha256 -ne $ExpectedSourceSha256 -or
        [string]$Authorization.model_sha256 -ne $ExpectedModelSha256 -or
        [string]$Authorization.ai_profiles_sha256 -ne $ExpectedAiProfilesSha256 -or
        [string]$Authorization.machine_type -ne 'c4-standard-4' -or
        -not (Test-M43A7Integer $Authorization.native_batch_threads) -or
        [int]$Authorization.native_batch_threads -ne 4 -or
        -not (Test-M43A7Integer $Authorization.jobs) -or [int]$Authorization.jobs -ne 5 -or
        $Authorization.spot_authorized -isnot [bool] -or $Authorization.spot_authorized -ne $true -or
        [string]$Authorization.actual_scalar_batch_result -ne 'pending_spot_preflight_receive_and_aggregate' -or
        [string]$Authorization.actual_operational_go_no_go -ne 'pending_spot_preflight_receive_and_aggregate') {
        throw 'Attempt07 remote launch authorization changed'
    }
    foreach ($name in @(
        'new_root_generation_allowed', 'arm_selection_allowed', 'current_profile_resolved',
        'current_profile_mutated', 'runtime_policy_activated'
    )) {
        Assert-M43A7False -Value $Authorization.$name -Label "authorization $name"
    }
    foreach ($name in @('correctness_smoke', 'determinism', 'scalar_batch_parity_test_harness', 'package_closure')) {
        if ([string]$Authorization.local_gates.$name -ne 'pass') {
            throw "Attempt07 remote authorization gate changed: $name"
        }
    }
}

function Assert-M43A7Done {
    param($Done, $Spec, $Manifest, [string]$ManifestSha256, [string]$AuthorizationSha256)
    $actualFields = @($Done.PSObject.Properties.Name | Sort-Object)
    if (($actualFields -join ',') -cne ($ExpectedDoneFields -join ',')) {
        throw 'Attempt07 DONE field set changed'
    }
    foreach ($name in @('job_index', 'source_root_index', 'native_batch_threads', 'peak_rss_bytes')) {
        if (-not (Test-M43A7Integer $Done.$name)) { throw "Attempt07 DONE $name must be an integer" }
    }
    if ($Done.batch_child_selectors -isnot [bool]) {
        throw 'Attempt07 DONE batch_child_selectors must be boolean'
    }
    foreach ($name in @(
        'teacher_values_exported', 'arm_selection_performed',
        'new_root_generated', 'current_profile_resolved', 'current_profile_mutated',
        'runtime_policy_activated'
    )) {
        Assert-M43A7False -Value $Done.$name -Label "DONE $name"
    }
    $elapsed = $Done.elapsed_seconds
    if ($elapsed -is [bool] -or $elapsed -isnot [ValueType] -or
        [double]::IsNaN([double]$elapsed) -or [double]::IsInfinity([double]$elapsed) -or
        [double]$elapsed -lt 0.0 -or [int64]$Done.peak_rss_bytes -lt 0) {
        throw 'Attempt07 DONE resource metrics changed'
    }
    if ($Done.schema -ne 'hu_m43_attempt07_preflight_spot_done_v1' -or
        $Done.status -ne 'complete' -or
        [string]$Done.run_name -ne $RunName -or
        [int]$Done.job_index -ne [int]$Spec.job_index -or
        [string]$Done.job_id -ne [string]$Spec.job_id -or
        [int]$Done.source_root_index -ne [int]$Spec.source_root_index -or
        [int]$Done.native_batch_threads -ne 4 -or
        [string]$Done.output_prefix -ne [string]$Spec.output_prefix -or
        [string]$Done.manifest_sha256 -ne $ManifestSha256 -or
        [string]$Done.authorization_sha256 -ne $AuthorizationSha256 -or
        [string]$Done.schedule_sha256 -ne [string]$Manifest.schedule_sha256 -or
        [string]$Done.attempt07_plan_sha256 -ne $ExpectedAttempt07PlanSha256 -or
        [string]$Done.source_merged_sha256 -ne $ExpectedSourceSha256 -or
        [string]$Done.model_sha256 -ne $ExpectedModelSha256 -or
        [string]$Done.ai_profiles_sha256 -ne $ExpectedAiProfilesSha256) {
        throw "Attempt07 DONE identity changed for job $($Spec.job_index)"
    }
    if ($Done.batch_child_selectors -ne [bool]$Spec.batch_child_selectors) {
        throw "Attempt07 DONE mode changed for job $($Spec.job_index)"
    }
    foreach ($name in @(
        'output_sha256', 'checkpoint_sha256', 'heartbeat_sha256',
        'summary_sha256', 'run_log_sha256'
    )) {
        Assert-M43A4Sha256 ([string]$Done.$name) "Attempt07 DONE $name"
    }
}

if ($RunName -notmatch '^[a-z0-9][a-z0-9-]{2,126}[a-z0-9]$') { throw 'unsafe RunName' }
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $RunDir) { $RunDir = Join-Path (Join-Path $repoRoot 'outputs/gcp_runs') $RunName }
$RunDir = Resolve-M43A4Path -Path $RunDir -Root $repoRoot -Label 'Attempt07 run directory'
Assert-M43A4UnderRoot -Path $RunDir -Root $repoRoot -Label 'Attempt07 run directory'
$manifestPath = Join-Path $RunDir 'manifest.json'
$schedulePath = Join-Path $RunDir 'shards_manifest.jsonl'
foreach ($path in @($manifestPath, $schedulePath)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Attempt07 closure missing: $path" }
}
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$manifestSha256 = Get-M43A4Sha256 $manifestPath
if ($manifest.schema -ne 'hu_m43_attempt07_preflight_spot_package_manifest_v1' -or
    $manifest.status -ne 'packaged_attempt06_copy_plus_overlay_without_execution' -or
    [string]$manifest.run_name -ne $RunName -or [int]$manifest.jobs -ne $JobCount -or
    [string]$manifest.machine_type -ne 'c4-standard-4' -or
    [int]$manifest.native_batch_threads -ne 4 -or
    [string]$manifest.schedule_sha256 -ne (Get-M43A4Sha256 $schedulePath) -or
    [string]$manifest.attempt07_plan_sha256 -ne $ExpectedAttempt07PlanSha256 -or
    [string]$manifest.source_merged_sha256 -ne $ExpectedSourceSha256 -or
    [string]$manifest.model_sha256 -ne $ExpectedModelSha256 -or
    [string]$manifest.ai_profiles_sha256 -ne $ExpectedAiProfilesSha256) {
    throw 'Attempt07 local package closure changed'
}
foreach ($name in @(
    'new_root_generated', 'teacher_executed', 'gcloud_invoked', 'instances_created',
    'arm_selection_performed', 'current_profile_resolved', 'current_profile_mutated',
    'runtime_policy_activated'
)) {
    Assert-M43A7False -Value $manifest.$name -Label "manifest $name"
}

$specs = @([IO.File]::ReadLines($schedulePath) | Where-Object { $_.Trim() } | ForEach-Object { $_ | ConvertFrom-Json })
$expectedIds = @('root0_batch_a', 'root0_batch_b', 'root0_scalar', 'root1_batch', 'root2_batch')
$expectedRoots = @(0, 0, 0, 1, 2)
$expectedBatch = @($true, $true, $false, $true, $true)
if ($specs.Count -ne $JobCount) { throw 'Attempt07 schedule must contain exactly five jobs' }
for ($index = 0; $index -lt $JobCount; $index++) {
    $spec = $specs[$index]
    if (-not (Test-M43A7Integer $spec.job_index) -or
        -not (Test-M43A7Integer $spec.source_root_index) -or
        -not (Test-M43A7Integer $spec.native_batch_threads) -or
        $spec.batch_child_selectors -isnot [bool] -or
        [int]$spec.job_index -ne $index -or [string]$spec.job_id -ne $expectedIds[$index] -or
        [int]$spec.source_root_index -ne $expectedRoots[$index] -or
        [bool]$spec.batch_child_selectors -ne $expectedBatch[$index] -or
        [int]$spec.native_batch_threads -ne 4 -or [string]$spec.machine_type -ne 'c4-standard-4' -or
        [string]$spec.output_prefix -ne "job_$($index.ToString('000'))_$($expectedIds[$index])") {
        throw "Attempt07 schedule identity changed for job $index"
    }
    Assert-M43A7False -Value $spec.new_root_generation_allowed -Label "schedule job $index new_root_generation_allowed"
    Assert-M43A7False -Value $spec.arm_selection_allowed -Label "schedule job $index arm_selection_allowed"
}

$selected = @(Expand-M43A7StatusSelection -Values $Jobs)
if ($selected.Count -eq 0) { throw 'no Attempt07 status jobs selected' }
$prefix = "gs://$Bucket/runs/$RunName"
$temporaryRoot = Join-Path ([IO.Path]::GetTempPath()) ('m43a7-status-closure-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $temporaryRoot | Out-Null
try {
    Copy-M43A4RemoteFileExact -Uri "$prefix/manifest.json" `
        -Destination (Join-Path $temporaryRoot 'manifest.json') -ProjectId $ProjectId `
        -ExpectedSha256 $manifestSha256
    Copy-M43A4RemoteFileExact -Uri "$prefix/source/shards_manifest.jsonl" `
        -Destination (Join-Path $temporaryRoot 'shards_manifest.jsonl') -ProjectId $ProjectId `
        -ExpectedSha256 ([string]$manifest.schedule_sha256)
    $remoteAuthorizationPath = Join-Path $temporaryRoot 'spot_authorization.json'
    Copy-M43A4RemoteFileExact -Uri "$prefix/source/spot_authorization.json" `
        -Destination $remoteAuthorizationPath -ProjectId $ProjectId
    $remoteAuthorization = Get-Content -LiteralPath $remoteAuthorizationPath -Raw | ConvertFrom-Json
    Assert-M43A7Authorization `
        -Authorization $remoteAuthorization -ManifestSha256 $manifestSha256 -Manifest $manifest
    $authorizationSha256 = Get-M43A4Sha256 $remoteAuthorizationPath
}
finally { Remove-Item -LiteralPath $temporaryRoot -Recurse -Force -ErrorAction SilentlyContinue }

# Status reads exact DONE metadata only.  Proof, log, checkpoint, heartbeat and
# summary contents remain unopened; only exact-object existence is queried.
$rows = foreach ($jobIndex in $selected) {
    $spec = $specs[$jobIndex]
    $resultPrefix = "$prefix/results/$($spec.output_prefix)"
    $objectUris = [ordered]@{
        proof = "$resultPrefix/preflight.json"
        checkpoint = "$resultPrefix/checkpoint.json"
        heartbeat = "$resultPrefix/heartbeat.json"
        summary = "$resultPrefix/summary.json"
        log = "$resultPrefix/run.log"
        done = "$resultPrefix/DONE.json"
    }
    $presence = @{}
    foreach ($name in $objectUris.Keys) {
        $presence[$name] = Test-M43A4GcsObject -Uri $objectUris[$name] -ProjectId $ProjectId
    }
    if ($presence.done) {
        $done = Read-M43A7DoneExact -Uri $objectUris.done
        Assert-M43A7Done `
            -Done $done -Spec $spec -Manifest $manifest `
            -ManifestSha256 $manifestSha256 -AuthorizationSha256 $authorizationSha256
        foreach ($name in @('proof', 'checkpoint', 'heartbeat', 'summary', 'log')) {
            if (-not $presence[$name]) { throw "Attempt07 DONE exists but $name is missing for job $jobIndex" }
        }
    }
    [pscustomobject][ordered]@{
        job_index = $jobIndex
        job_id = [string]$spec.job_id
        source_root_index = [int]$spec.source_root_index
        mode = [string]$spec.mode
        state = $(if ($presence.done) { 'complete_verified_done' } else { 'not_complete' })
        proof_exists = [bool]$presence.proof
        checkpoint_exists = [bool]$presence.checkpoint
        heartbeat_exists = [bool]$presence.heartbeat
        summary_exists = [bool]$presence.summary
        log_exists = [bool]$presence.log
        done_exists = [bool]$presence.done
    }
}
$complete = @($rows | Where-Object { $_.state -eq 'complete_verified_done' }).Count
[pscustomobject][ordered]@{
    schema = 'hu_m43_attempt07_preflight_spot_status_result_v1'
    run_name = $RunName
    selected_jobs = $selected.Count
    complete_jobs = $complete
    incomplete_jobs = $selected.Count - $complete
    all_done_markers_verified = ($complete -eq $selected.Count)
    manifest_sha256 = $manifestSha256
    schedule_sha256 = [string]$manifest.schedule_sha256
    authorization_sha256 = $authorizationSha256
    result_payload_downloaded = $false
    log_payload_downloaded = $false
    checkpoint_payload_downloaded = $false
    heartbeat_payload_downloaded = $false
    exact_done_objects_only = $true
    jobs = $rows
    current_profile_resolved = $false
    current_profile_mutated = $false
} | ConvertTo-Json -Depth 8
