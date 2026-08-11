param(
    [Parameter(Mandatory = $true)][string]$RunName,
    [string]$ProjectId = 'ofc-solver-485418',
    [string]$Bucket = 'pokerhu-ofc-solver-485418-training',
    [string]$RunDir,
    [string]$OutputDir,
    [string]$SourcePath = 'outputs/hu_joint_policy/m43_attempt06_search_quality/regular-hu-m43-attempt06-preflight-final-20260714-1154/merged/teacher.jsonl'
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

function Invoke-M43A7ReceivePython {
    param([string[]]$Arguments, [string]$Label, [int]$TimeoutSeconds = 300)
    $pythonPath = (Get-Command python -ErrorAction Stop | Select-Object -First 1).Source
    $result = Invoke-M43A4ProcessBounded `
        -FilePath $pythonPath -Arguments $Arguments -Label $Label `
        -TimeoutSeconds $TimeoutSeconds `
        -Environment @{ PYTHONPATH = (Join-Path $script:RepoRoot 'src') }
    if ($result.timed_out -or $result.exit_code -ne 0) {
        throw "$Label failed: $($result.stderr)$($result.stdout)"
    }
    return [string]$result.stdout
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

function Assert-M43A7ReceiveDone {
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
        'teacher_values_exported', 'arm_selection_performed', 'new_root_generated',
        'current_profile_resolved', 'current_profile_mutated', 'runtime_policy_activated'
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
        $Done.status -ne 'complete' -or [string]$Done.run_name -ne $RunName -or
        [int]$Done.job_index -ne [int]$Spec.job_index -or
        [string]$Done.job_id -ne [string]$Spec.job_id -or
        [int]$Done.source_root_index -ne [int]$Spec.source_root_index -or
        $Done.batch_child_selectors -ne [bool]$Spec.batch_child_selectors -or
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
    foreach ($name in @(
        'output_sha256', 'checkpoint_sha256', 'heartbeat_sha256',
        'summary_sha256', 'run_log_sha256'
    )) {
        Assert-M43A4Sha256 ([string]$Done.$name) "Attempt07 DONE $name"
    }
}

function Publish-M43A7ReceiveDirectoryAtomic {
    param([string]$Source, [string]$Destination)
    $sourceFull = [IO.Path]::GetFullPath($Source)
    $destinationFull = [IO.Path]::GetFullPath($Destination)
    if (Test-Path -LiteralPath $destinationFull) {
        throw "Attempt07 receive destination already exists: $destinationFull"
    }
    [IO.Directory]::Move($sourceFull, $destinationFull)
}

if ($RunName -notmatch '^[a-z0-9][a-z0-9-]{2,126}[a-z0-9]$') { throw 'unsafe RunName' }
$script:RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $RunDir) { $RunDir = Join-Path (Join-Path $script:RepoRoot 'outputs/gcp_runs') $RunName }
if (-not $OutputDir) {
    $OutputDir = Join-Path $script:RepoRoot "outputs/hu_joint_policy/m43_attempt07_preflight/$RunName"
}
$RunDir = Resolve-M43A4Path -Path $RunDir -Root $script:RepoRoot -Label 'Attempt07 run directory'
$OutputDir = Resolve-M43A4Path -Path $OutputDir -Root $script:RepoRoot -Label 'Attempt07 receive output'
$resolvedSource = Resolve-M43A4Path `
    -Path $SourcePath -Root $script:RepoRoot -Label 'Attempt07 immutable preflight source' -RequireFile
Assert-M43A4UnderRoot -Path $RunDir -Root $script:RepoRoot -Label 'Attempt07 run directory'
Assert-M43A4UnderRoot -Path $OutputDir -Root $script:RepoRoot -Label 'Attempt07 receive output'
if (Test-Path -LiteralPath $OutputDir) {
    throw 'Attempt07 receive output already exists; immutable output is never overwritten'
}
if ((Get-M43A4Sha256 $resolvedSource) -ne $ExpectedSourceSha256) {
    throw 'Attempt07 immutable preflight source changed before aggregate'
}

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
    throw 'Attempt07 local receive closure changed'
}
foreach ($name in @(
    'new_root_generated', 'teacher_executed', 'gcloud_invoked', 'instances_created',
    'arm_selection_performed', 'current_profile_resolved', 'current_profile_mutated',
    'runtime_policy_activated'
)) {
    Assert-M43A7False -Value $manifest.$name -Label "manifest $name"
}
$specs = @([IO.File]::ReadLines($schedulePath) | Where-Object { $_.Trim() } | ForEach-Object { $_ | ConvertFrom-Json })
if ($specs.Count -ne $JobCount) { throw 'Attempt07 receive requires exactly five jobs' }
for ($index = 0; $index -lt $JobCount; $index++) {
    if ([int]$specs[$index].job_index -ne $index -or
        [int]$specs[$index].source_root_index -notin @(0, 1, 2) -or
        [string]$specs[$index].machine_type -ne 'c4-standard-4' -or
        [int]$specs[$index].native_batch_threads -ne 4) {
        throw "Attempt07 receive schedule changed for job $index"
    }
}

$prefix = "gs://$Bucket/runs/$RunName"
$temporaryRoot = Join-Path ([IO.Path]::GetTempPath()) ('m43a7-receive-closure-' + [guid]::NewGuid().ToString('N'))
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

$stagingRoot = $OutputDir + '.staging-' + [guid]::NewGuid().ToString('N')
try {
    New-Item -ItemType Directory -Path $stagingRoot | Out-Null
    $closureDir = Join-Path $stagingRoot 'closure'
    $jobsDir = Join-Path $stagingRoot 'jobs'
    $mergedDir = Join-Path $stagingRoot 'merged'
    New-Item -ItemType Directory -Path $closureDir, $jobsDir, $mergedDir | Out-Null
    Copy-Item -LiteralPath $manifestPath -Destination (Join-Path $closureDir 'manifest.json')
    Copy-Item -LiteralPath $schedulePath -Destination (Join-Path $closureDir 'shards_manifest.jsonl')
    $receivedAuthorizationPath = Join-Path $closureDir 'spot_authorization.json'
    Copy-M43A4RemoteFileExact -Uri "$prefix/source/spot_authorization.json" `
        -Destination $receivedAuthorizationPath -ProjectId $ProjectId `
        -ExpectedSha256 $authorizationSha256
    foreach ($name in @(
        'hu_joint_policy_m43_attempt07_preflight.json',
        'hu_joint_policy_m43_attempt07.json',
        'startup_hu_m43_attempt07_preflight.sh'
    )) {
        $source = Join-Path $RunDir $name
        if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
            throw "Attempt07 receive package closure is missing: $source"
        }
        Copy-Item -LiteralPath $source -Destination (Join-Path $closureDir $name)
    }

    foreach ($spec in $specs) {
        $jobDir = Join-Path $jobsDir ([string]$spec.output_prefix)
        New-Item -ItemType Directory -Path $jobDir | Out-Null
        $resultPrefix = "$prefix/results/$($spec.output_prefix)"
        $donePath = Join-Path $jobDir 'DONE.json'
        Copy-M43A4RemoteFileExact `
            -Uri "$resultPrefix/DONE.json" -Destination $donePath -ProjectId $ProjectId
        $done = Get-Content -LiteralPath $donePath -Raw | ConvertFrom-Json
        Assert-M43A7ReceiveDone `
            -Done $done -Spec $spec -Manifest $manifest `
            -ManifestSha256 $manifestSha256 -AuthorizationSha256 $authorizationSha256
        foreach ($binding in @(
            @('preflight.json', 'output_sha256'),
            @('checkpoint.json', 'checkpoint_sha256'),
            @('heartbeat.json', 'heartbeat_sha256'),
            @('summary.json', 'summary_sha256'),
            @('run.log', 'run_log_sha256')
        )) {
            $name = [string]$binding[0]
            $hashField = [string]$binding[1]
            Copy-M43A4RemoteFileExact `
                -Uri "$resultPrefix/$name" -Destination (Join-Path $jobDir $name) `
                -ProjectId $ProjectId -ExpectedSha256 ([string]$done.$hashField)
        }
        $auditPath = Join-Path $jobDir 'received_audit.json'
        $auditRaw = Invoke-M43A7ReceivePython `
            -Arguments @(
                '-B', '-m', 'ofc_regular.hu_m43_attempt07_preflight_spot',
                'validate-received-job', '--job-dir', $jobDir,
                '--schedule', (Join-Path $closureDir 'shards_manifest.jsonl'),
                '--manifest', (Join-Path $closureDir 'manifest.json'),
                '--authorization', $receivedAuthorizationPath,
                '--job-index', ([string]$spec.job_index), '--output', $auditPath
            ) -Label "validate received Attempt07 job $($spec.job_index)"
        $audit = $auditRaw.Trim() | ConvertFrom-Json
        if ($audit.schema -ne 'hu_m43_attempt07_preflight_receive_audit_v1' -or
            $audit.status -ne 'validated_value_redacted_preflight_proof' -or
            [int]$audit.job_index -ne [int]$spec.job_index) {
            throw "Attempt07 receive audit failed for job $($spec.job_index)"
        }
    }

    $mergeRaw = Invoke-M43A7ReceivePython `
        -Arguments @(
            '-B', '-m', 'ofc_regular.hu_m43_attempt07_preflight_spot',
            'merge-received-jobs', '--jobs-root', $jobsDir,
            '--schedule', (Join-Path $closureDir 'shards_manifest.jsonl'),
            '--manifest', (Join-Path $closureDir 'manifest.json'),
            '--output', (Join-Path $mergedDir 'preflight.jsonl'),
            '--receipt', (Join-Path $mergedDir 'merge_receipt.json')
        ) -Label 'merge exactly five Attempt07 value-redacted preflight proofs'
    $merge = $mergeRaw.Trim() | ConvertFrom-Json
    if ($merge.schema -ne 'hu_m43_attempt07_preflight_receive_merge_v1' -or
        $merge.status -ne 'merged_five_value_redacted_preflight_proofs' -or
        [int]$merge.jobs -ne $JobCount) {
        throw 'Attempt07 receive merge boundary changed'
    }
    foreach ($name in @('teacher_values_opened', 'arm_selection_performed', 'current_profile_mutated')) {
        Assert-M43A7False -Value $merge.$name -Label "merge $name"
    }
    $aggregatePath = Join-Path $mergedDir 'preflight_aggregate.json'
    $aggregateArguments = @(
        '-B', '-m', 'ofc_regular.aggregate_hu_m43_attempt07_preflight',
        '--output', $aggregatePath,
        '--source', $resolvedSource,
        '--preflight-plan', (Join-Path $closureDir 'hu_joint_policy_m43_attempt07_preflight.json'),
        '--attempt07-plan', (Join-Path $closureDir 'hu_joint_policy_m43_attempt07.json')
    )
    foreach ($spec in $specs) {
        $jobDir = Join-Path $jobsDir ([string]$spec.output_prefix)
        $slotFlag = '--' + ([string]$spec.job_id -replace '_', '-')
        $aggregateArguments += @($slotFlag, (Join-Path $jobDir 'preflight.json'))
        $aggregateArguments += @(
            '--done-metadata',
            "$($spec.job_id)=$(Join-Path $jobDir 'DONE.json')"
        )
    }
    [void](Invoke-M43A7ReceivePython `
        -Arguments $aggregateArguments `
        -Label 'aggregate five Attempt07 proofs with separate DONE diagnostics')
    $aggregate = Get-Content -LiteralPath $aggregatePath -Raw | ConvertFrom-Json
    if ($aggregate.schema -ne 'hu_m43_attempt07_preflight_proof_aggregate_v1' -or
        $aggregate.status -ne 'complete' -or
        [string]$aggregate.decision -notin @('go', 'no_go') -or
        [int]$aggregate.proof_input_count -ne $JobCount -or
        $aggregate.science_boundary.done_metadata_is_science_input -isnot [bool] -or
        $aggregate.science_boundary.done_metadata_is_science_input -ne $false -or
        [string]$aggregate.operational_diagnostics.status -ne 'ok' -or
        [int]$aggregate.operational_diagnostics.job_count -ne $JobCount -or
        $aggregate.operational_diagnostics.science_decision_input -isnot [bool] -or
        $aggregate.operational_diagnostics.science_decision_input -ne $false) {
        throw 'Attempt07 proof aggregate boundary changed'
    }
    $receipt = [ordered]@{
        schema = 'hu_m43_attempt07_preflight_spot_receive_receipt_v1'
        status = 'verified_five_value_redacted_preflight_jobs'
        run_name = $RunName
        manifest_sha256 = $manifestSha256
        schedule_sha256 = [string]$manifest.schedule_sha256
        authorization_sha256 = $authorizationSha256
        preflight_plan_sha256 = [string]$manifest.preflight_plan_sha256
        merged_preflight_sha256 = Get-M43A4Sha256 (Join-Path $mergedDir 'preflight.jsonl')
        proof_aggregate_sha256 = Get-M43A4Sha256 $aggregatePath
        decision = [string]$aggregate.decision
        jobs = $JobCount
        source_roots = @(0, 1, 2)
        teacher_values_opened = $false
        arm_selection_performed = $false
        operational_authorization_computed = $false
        current_profile_resolved = $false
        current_profile_mutated = $false
        runtime_policy_activated = $false
    }
    Write-M43A4Utf8CreateNew `
        -Path (Join-Path $mergedDir 'receive_receipt.json') `
        -Text (($receipt | ConvertTo-Json -Depth 6) + "`n")
    Publish-M43A7ReceiveDirectoryAtomic -Source $stagingRoot -Destination $OutputDir
}
catch {
    if (Test-Path -LiteralPath $stagingRoot) {
        Remove-Item -LiteralPath $stagingRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
    throw
}

[pscustomobject][ordered]@{
    schema = 'hu_m43_attempt07_preflight_spot_receive_result_v1'
    status = 'published_exact_five_job_preflight'
    run_name = $RunName
    output_dir = $OutputDir
    merged_preflight = Join-Path $OutputDir 'merged/preflight.jsonl'
    manifest_sha256 = $manifestSha256
    authorization_sha256 = $authorizationSha256
    proof_aggregate_sha256 = Get-M43A4Sha256 (Join-Path $OutputDir 'merged/preflight_aggregate.json')
    decision = [string]$aggregate.decision
    jobs = $JobCount
    source_roots = @(0, 1, 2)
    teacher_values_opened = $false
    arm_selection_performed = $false
    operational_authorization_computed = $false
    current_profile_resolved = $false
    current_profile_mutated = $false
} | ConvertTo-Json -Depth 6
