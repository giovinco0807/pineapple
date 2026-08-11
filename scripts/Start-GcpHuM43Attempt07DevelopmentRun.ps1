param(
    [string]$ProjectId = 'ofc-solver-485418',
    [string]$Bucket = 'pokerhu-ofc-solver-485418-training',
    [string]$RunName = ('regular-hu-m43-attempt07-development100-' + (Get-Date -Format 'yyyyMMdd-HHmmss')),
    [string]$PlanPath = 'configs/hu_joint_policy_m43_attempt07.json',
    [string]$StatusPath = 'configs/hu_joint_policy_m43_attempt07_status.json',
    [string]$PreflightPlanPath = 'configs/hu_joint_policy_m43_attempt07_preflight.json',
    [string]$PreflightAggregatePath,
    [string]$PreflightReceiveDir,
    [string]$ModelPath = 'outputs/hu_joint_policy/m43_attempt05_old_dev900_architecture_once/lambda_rank_candidate.pkl',
    [string]$SpotAuthorizationPath,
    [string]$RunDir,
    [string[]]$StartShards = @('0-4'),
    [string]$Zone = 'asia-northeast1-b',
    [string]$MachineType = 'c4-standard-4',
    [string]$BootDiskType = 'hyperdisk-balanced',
    [ValidateRange(20, 100)][int]$BootDiskGb = 50,
    [ValidateRange(10, 600)][int]$SyncIntervalSeconds = 60,
    [switch]$PackageOnly,
    [switch]$CreateInstances,
    [switch]$ResumePackage,
    [switch]$ResumeExisting,
    [switch]$SkipExistingInstances,
    [switch]$NoSelfDelete
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'HuM43Attempt04Spot.Common.ps1')

$ExpectedPlanSha256 = '8bf15f8d109a2e441e1c240e533c56df42a7e65d2eefd3b2750d1bd63758b189'
$ExpectedModelSha256 = 'e7fa728308c8973e2e202baf79309e30b9b6f58c37431d8ea55850390abc28c3'
$ExpectedAiProfilesSha256 = 'd2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3'
$TotalShards = 100
$MaxWaveShards = 50
$NativeBatchThreads = 4

if ($RunName -notmatch '^[a-z0-9][a-z0-9-]{2,126}[a-z0-9]$') {
    throw 'RunName is not a safe GCP run identity'
}
if ($PackageOnly -and $CreateInstances) {
    throw 'PackageOnly and CreateInstances are mutually exclusive'
}
if (-not $PackageOnly -and -not $CreateInstances) {
    throw 'Choose PackageOnly or explicitly choose CreateInstances'
}
if ($MachineType -ne 'c4-standard-4') {
    throw 'Attempt07 development is frozen to c4-standard-4'
}

function Expand-M43A7ShardSelection {
    param([string[]]$Values)
    $selected = [Collections.Generic.SortedSet[int]]::new()
    foreach ($raw in $Values) {
        foreach ($token in ([string]$raw -split ',')) {
            $value = $token.Trim()
            if (-not $value) { continue }
            if ($value -match '^(\d+)-(\d+)$') {
                $first = [int]$Matches[1]
                $last = [int]$Matches[2]
                if ($first -gt $last) { throw "Invalid shard range: $value" }
                for ($index = $first; $index -le $last; $index++) {
                    [void]$selected.Add($index)
                }
            }
            elseif ($value -match '^\d+$') { [void]$selected.Add([int]$value) }
            else { throw "Invalid shard selection: $value" }
        }
    }
    foreach ($index in $selected) {
        if ($index -lt 0 -or $index -ge $TotalShards) {
            throw "Shard index outside 0..99: $index"
        }
    }
    $result = @($selected)
    if ($result.Count -eq 0) { throw 'StartShards selected no shards' }
    if ($result.Count -gt $MaxWaveShards) {
        throw 'One launch wave may contain at most 50 shards'
    }
    return $result
}

function ConvertTo-M43A7VmPrefix {
    param([string]$Value)
    $result = (($Value.ToLowerInvariant() -replace '[^a-z0-9-]', '-') -replace '-+', '-').Trim('-')
    if (-not $result) { throw 'RunName cannot produce a VM prefix' }
    if ($result.Length -gt 54) { $result = $result.Substring(0, 54).TrimEnd('-') }
    return $result
}

function Invoke-M43A7Python {
    param([string[]]$Arguments, [string]$Label, [int]$TimeoutSeconds = 1200)
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

$script:RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $RunDir) {
    $RunDir = Join-Path (Join-Path $script:RepoRoot 'outputs/gcp_runs') $RunName
}
if (-not $PreflightAggregatePath) {
    throw 'PreflightAggregatePath is required and must be the immutable Go aggregate'
}
if (-not $PreflightReceiveDir) {
    throw 'PreflightReceiveDir is required to freeze the exact five-job preflight closure'
}
$RunDir = Resolve-M43A4Path -Path $RunDir -Root $script:RepoRoot -Label 'Attempt07 run directory'
Assert-M43A4UnderRoot -Path $RunDir -Root $script:RepoRoot -Label 'Attempt07 run directory'
$expectedRunDir = [IO.Path]::GetFullPath(
    (Join-Path (Join-Path $script:RepoRoot 'outputs/gcp_runs') $RunName)
).TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar)
$actualRunDir = [IO.Path]::GetFullPath($RunDir).TrimEnd(
    [IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar
)
if (-not [string]::Equals($actualRunDir, $expectedRunDir, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Attempt07 RunDir must be outputs/gcp_runs/<RunName>'
}
$plan = Resolve-M43A4Path -Path $PlanPath -Root $script:RepoRoot -Label 'Attempt07 plan' -RequireFile
$status = Resolve-M43A4Path -Path $StatusPath -Root $script:RepoRoot -Label 'Attempt07 status' -RequireFile
$preflightPlan = Resolve-M43A4Path -Path $PreflightPlanPath -Root $script:RepoRoot -Label 'Attempt07 preflight plan' -RequireFile
$preflightAggregate = Resolve-M43A4Path -Path $PreflightAggregatePath -Root $script:RepoRoot -Label 'Attempt07 preflight aggregate' -RequireFile
$preflightReceive = Resolve-M43A4Path -Path $PreflightReceiveDir -Root $script:RepoRoot -Label 'Attempt07 preflight receive directory'
Assert-M43A4UnderRoot -Path $preflightReceive -Root $script:RepoRoot -Label 'Attempt07 preflight receive directory'
$preflightManifest = Resolve-M43A4Path -Path (Join-Path $preflightReceive 'closure/manifest.json') -Root $script:RepoRoot -Label 'Attempt07 preflight manifest' -RequireFile
$preflightSchedule = Resolve-M43A4Path -Path (Join-Path $preflightReceive 'closure/shards_manifest.jsonl') -Root $script:RepoRoot -Label 'Attempt07 preflight schedule' -RequireFile
$preflightLaunchAuthorization = Resolve-M43A4Path -Path (Join-Path $preflightReceive 'closure/spot_authorization.json') -Root $script:RepoRoot -Label 'Attempt07 preflight launch authorization' -RequireFile
$preflightDone = [ordered]@{
    root0_batch_a = Resolve-M43A4Path -Path (Join-Path $preflightReceive 'jobs/job_000_root0_batch_a/DONE.json') -Root $script:RepoRoot -Label 'preflight root0 batch A DONE' -RequireFile
    root0_batch_b = Resolve-M43A4Path -Path (Join-Path $preflightReceive 'jobs/job_001_root0_batch_b/DONE.json') -Root $script:RepoRoot -Label 'preflight root0 batch B DONE' -RequireFile
    root0_scalar = Resolve-M43A4Path -Path (Join-Path $preflightReceive 'jobs/job_002_root0_scalar/DONE.json') -Root $script:RepoRoot -Label 'preflight root0 scalar DONE' -RequireFile
    root1_batch = Resolve-M43A4Path -Path (Join-Path $preflightReceive 'jobs/job_003_root1_batch/DONE.json') -Root $script:RepoRoot -Label 'preflight root1 batch DONE' -RequireFile
    root2_batch = Resolve-M43A4Path -Path (Join-Path $preflightReceive 'jobs/job_004_root2_batch/DONE.json') -Root $script:RepoRoot -Label 'preflight root2 batch DONE' -RequireFile
}
$model = Resolve-M43A4Path -Path $ModelPath -Root $script:RepoRoot -Label 'Attempt07 model' -RequireFile
$startup = Resolve-M43A4Path -Path (Join-Path $PSScriptRoot 'startup_hu_m43_attempt07_development.sh') -Root $script:RepoRoot -Label 'Attempt07 startup' -RequireFile
if ((Get-M43A4Sha256 $plan) -ne $ExpectedPlanSha256) { throw 'Attempt07 plan SHA changed' }
if ((Get-M43A4Sha256 $model) -ne $ExpectedModelSha256) { throw 'Attempt07 model SHA changed' }
if ((Get-M43A4Sha256 (Join-Path $script:RepoRoot 'src/ofc_regular/ai_profiles.py')) -ne $ExpectedAiProfilesSha256) {
    throw 'Attempt07 ai_profiles.py SHA changed'
}

$packageArgs = @(
    '-B', '-m', 'ofc_regular.hu_m43_attempt07_spot', 'package',
    '--repo-root', $script:RepoRoot, '--run-dir', $RunDir, '--run-name', $RunName,
    '--plan', $plan, '--status', $status, '--model', $model,
    '--startup', $startup, '--preflight-aggregate', $preflightAggregate,
    '--preflight-plan', $preflightPlan,
    '--preflight-manifest', $preflightManifest,
    '--preflight-schedule', $preflightSchedule,
    '--preflight-launch-authorization', $preflightLaunchAuthorization,
    '--preflight-done-root0-batch-a', $preflightDone.root0_batch_a,
    '--preflight-done-root0-batch-b', $preflightDone.root0_batch_b,
    '--preflight-done-root0-scalar', $preflightDone.root0_scalar,
    '--preflight-done-root1-batch', $preflightDone.root1_batch,
    '--preflight-done-root2-batch', $preflightDone.root2_batch
)
if (Test-Path -LiteralPath $RunDir) {
    if (-not $ResumePackage -and -not $CreateInstances) {
        throw 'Attempt07 package exists; use ResumePackage to validate and reuse it'
    }
    $packageArgs += '--resume-existing'
}
$packageRaw = Invoke-M43A7Python -Arguments $packageArgs -Label 'package Attempt07 without opening a root'
$packageResult = $packageRaw.Trim() | ConvertFrom-Json
if ($packageResult.schema -ne 'hu_m43_attempt07_development_package_result_v1' -or
    $packageResult.status -ne 'packaged_without_root_or_gcloud' -or
    [int]$packageResult.total_shards -ne 100 -or
    $packageResult.fresh_root_opened -ne $false -or
    $packageResult.teacher_executed -ne $false -or
    $packageResult.gcloud_invoked -ne $false -or
    $packageResult.instances_created -ne $false -or
    $packageResult.current_profile_mutated -ne $false) {
    throw 'Attempt07 package-only boundary changed'
}

# PackageOnly terminates before resolving gcloud, reading seed/root content, or
# publishing anything.  It is the required first command for every new run.
if ($PackageOnly) {
    $packageResult | ConvertTo-Json -Depth 8
    return
}

if (-not $SpotAuthorizationPath) {
    throw 'SpotAuthorizationPath is required for CreateInstances after package finalization'
}
$authorizationPath = Resolve-M43A4Path -Path $SpotAuthorizationPath -Root $script:RepoRoot -Label 'Attempt07 Spot authorization' -RequireFile
$manifestPath = Join-Path $RunDir 'manifest.json'
$schedulePath = Join-Path $RunDir 'shards_manifest.jsonl'
$aggregateCopy = Join-Path $RunDir 'attempt07_preflight_aggregate.json'
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$manifestSha256 = Get-M43A4Sha256 $manifestPath
if ($manifest.schema -ne 'hu_m43_attempt07_development_spot_package_v1' -or
    $manifest.status -ne 'frozen_package_only_no_root_opened' -or
    [string]$manifest.run_name -ne $RunName -or
    [int]$manifest.total_shards -ne 100 -or [int]$manifest.roots_per_shard -ne 1 -or
    [int]$manifest.native_batch_threads -ne 4 -or
    [string]$manifest.plan_sha256 -ne $ExpectedPlanSha256 -or
    [string]$manifest.model_sha256 -ne $ExpectedModelSha256 -or
    [string]$manifest.ai_profiles_sha256 -ne $ExpectedAiProfilesSha256 -or
    [string]$manifest.schedule_sha256 -ne (Get-M43A4Sha256 $schedulePath) -or
    [string]$manifest.preflight_aggregate_sha256 -ne (Get-M43A4Sha256 $aggregateCopy) -or
    $manifest.fresh_root_opened -ne $false -or $manifest.teacher_executed -ne $false -or
    $manifest.current_profile_mutated -ne $false -or $manifest.runtime_policy_activated -ne $false) {
    throw 'Attempt07 package manifest closure changed'
}
$authorizationRaw = Invoke-M43A7Python `
    -Arguments @(
        '-B', '-m', 'ofc_regular.hu_m43_attempt07_spot', 'validate-authorization',
        '--authorization', $authorizationPath, '--manifest', $manifestPath,
        '--preflight-aggregate', $aggregateCopy
    ) -Label 'validate immutable Attempt07 operational authorization'
$authorizationAudit = $authorizationRaw.Trim() | ConvertFrom-Json
if ($authorizationAudit.status -ne 'pass' -or $authorizationAudit.all_gates_passed -ne $true -or
    $authorizationAudit.development_started -ne $false -or
    $authorizationAudit.current_profile_mutated -ne $false -or
    $authorizationAudit.runtime_policy_activated -ne $false) {
    throw 'Attempt07 operational authorization did not pass'
}
$authorization = Get-Content -LiteralPath $authorizationPath -Raw | ConvertFrom-Json
$authorizationProperties = @($authorization.PSObject.Properties.Name)
$expectedAuthorizationProperties = @(
    'schema','status','spot_authorized','attempt07_plan_sha256','preflight_plan_sha256',
    'preflight_aggregate_sha256','preflight_manifest_sha256','preflight_schedule_sha256',
    'preflight_launch_authorization_sha256','done_sha256','operational_metrics',
    'development_run_name','development_manifest_sha256','development_schedule_sha256',
    'development_source_closure_sha256','development_source_zip_sha256',
    'development_startup_sha256','development_status_sha256',
    'development_source_model_manifest_sha256','development_source_native_manifest_sha256',
    'development_total_roots','development_total_shards','development_roots_per_shard',
    'development_native_batch_threads','development_max_wave_shards',
    'development_machine_type','development_root_profile_assignment',
    'development_batch_child_selectors','development_package_frozen_before_authorization',
    'operational_gates','all_gates_passed','development_started',
    'fresh_development_root_opened','current_profile_mutated','runtime_policy_activated'
)
if (@(Compare-Object $authorizationProperties $expectedAuthorizationProperties).Count -ne 0 -or
    $authorization.schema -ne 'hu_m43_attempt07_development_spot_authorization_v1' -or
    $authorization.status -ne 'authorized_after_attempt07_preflight' -or
    $authorization.spot_authorized -ne $true -or $authorization.all_gates_passed -ne $true -or
    [string]$authorization.attempt07_plan_sha256 -ne [string]$manifest.plan_sha256 -or
    [string]$authorization.preflight_plan_sha256 -ne [string]$manifest.preflight_plan_sha256 -or
    [string]$authorization.preflight_aggregate_sha256 -ne [string]$manifest.preflight_aggregate_sha256 -or
    [string]$authorization.development_run_name -ne $RunName -or
    [string]$authorization.development_manifest_sha256 -ne $manifestSha256 -or
    [string]$authorization.development_schedule_sha256 -ne [string]$manifest.schedule_sha256 -or
    [string]$authorization.development_source_closure_sha256 -ne [string]$manifest.source_closure_sha256 -or
    [string]$authorization.development_source_zip_sha256 -ne [string]$manifest.source_zip_sha256 -or
    [string]$authorization.development_startup_sha256 -ne [string]$manifest.startup_sha256 -or
    [string]$authorization.development_status_sha256 -ne [string]$manifest.status_sha256 -or
    [string]$authorization.development_source_model_manifest_sha256 -ne [string]$manifest.source_model_manifest_sha256 -or
    [string]$authorization.development_source_native_manifest_sha256 -ne [string]$manifest.source_native_manifest_sha256 -or
    [int]$authorization.development_total_roots -ne 100 -or
    [int]$authorization.development_total_shards -ne 100 -or
    [int]$authorization.development_roots_per_shard -ne 1 -or
    [int]$authorization.development_native_batch_threads -ne 4 -or
    [int]$authorization.development_max_wave_shards -ne 50 -or
    [string]$authorization.development_machine_type -ne 'c4-standard-4' -or
    [string]$authorization.development_root_profile_assignment -ne 'root_index_mod_5_in_frozen_profile_order' -or
    $authorization.development_batch_child_selectors -ne $true -or
    $authorization.development_package_frozen_before_authorization -ne $true -or
    $authorization.development_started -ne $false -or
    $authorization.fresh_development_root_opened -ne $false -or
    $authorization.current_profile_mutated -ne $false -or
    $authorization.runtime_policy_activated -ne $false) {
    throw 'Attempt07 Spot authorization exact identity changed'
}
$doneSlots = @($authorization.done_sha256.PSObject.Properties.Name)
if (@(Compare-Object $doneSlots @('root0_batch_a','root0_batch_b','root0_scalar','root1_batch','root2_batch')).Count -ne 0) {
    throw 'Attempt07 authorization does not bind exactly five preflight DONE files'
}
foreach ($property in $authorization.done_sha256.PSObject.Properties) {
    Assert-M43A4Sha256 ([string]$property.Value) "Attempt07 preflight DONE $($property.Name)"
}
Assert-M43A4Sha256 ([string]$authorization.preflight_launch_authorization_sha256) 'Attempt07 preflight launch authorization'
$authorizationSha256 = Get-M43A4Sha256 $authorizationPath
$selectedShards = @(Expand-M43A7ShardSelection -Values $StartShards)

function Publish-M43A7ImmutableObject {
    param([string]$Source, [string]$Uri)
    $upload = Invoke-M43A4GcloudProcess `
        -Arguments @('storage','cp',$Source,$Uri,'--project',$ProjectId,'--if-generation-match=0') `
        -TimeoutSeconds 600 -Label "publish immutable $Uri"
    if (-not $upload.timed_out -and $upload.exit_code -eq 0) { return }
    $temporary = Join-Path ([IO.Path]::GetTempPath()) ('m43a7-' + [guid]::NewGuid().ToString('N'))
    try {
        Copy-M43A4RemoteFileExact -Uri $Uri -Destination $temporary -ProjectId $ProjectId
        if ((Get-M43A4Sha256 $temporary) -ne (Get-M43A4Sha256 $Source)) {
            throw "Immutable remote object differs: $Uri"
        }
    }
    finally { Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue }
}

$sourcePath = Join-Path $RunDir 'ofc_regular_hu_m43_attempt07_development_source.zip'
$remotePackagePrefix = "gs://$Bucket/runs/$RunName/package"
$sourceUri = "$remotePackagePrefix/source.zip"
$manifestUri = "$remotePackagePrefix/manifest.json"
$authorizationUri = "$remotePackagePrefix/authorization-$authorizationSha256.json"
Publish-M43A7ImmutableObject -Source $sourcePath -Uri $sourceUri
Publish-M43A7ImmutableObject -Source $manifestPath -Uri $manifestUri
Publish-M43A7ImmutableObject -Source $authorizationPath -Uri $authorizationUri

$vmPrefix = ConvertTo-M43A7VmPrefix $RunName
$created = [Collections.Generic.List[int]]::new()
$skippedDone = [Collections.Generic.List[int]]::new()
$skippedExisting = [Collections.Generic.List[int]]::new()
foreach ($shard in $selectedShards) {
    $doneUri = "gs://$Bucket/runs/$RunName/results/shard_$('{0:D3}' -f $shard)/DONE.json"
    if (Test-M43A4GcsObject -Uri $doneUri -ProjectId $ProjectId) {
        $skippedDone.Add($shard)
        continue
    }
    $instanceName = "$vmPrefix-s$('{0:D3}' -f $shard)"
    $instanceExists = $false
    $describe = Invoke-M43A4GcloudProcess `
        -Arguments @('compute','instances','describe',$instanceName,'--zone',$Zone,'--project',$ProjectId,'--format=value(name)') `
        -TimeoutSeconds 30 -Label "describe $instanceName"
    if (-not $describe.timed_out -and $describe.exit_code -eq 0) { $instanceExists = $true }
    elseif (([string]$describe.stdout + [string]$describe.stderr) -notmatch '(?i)not found|was not found|404') {
        throw "Unable to prove instance state: $instanceName"
    }
    if ($instanceExists) {
        if ($SkipExistingInstances -or $ResumeExisting) {
            $skippedExisting.Add($shard)
            continue
        }
        throw "Instance already exists: $instanceName"
    }
    $metadata = @(
        "PROJECT_ID=$ProjectId", "BUCKET=$Bucket", "RUN_NAME=$RunName", "SHARD_INDEX=$shard",
        "SOURCE_URI=$sourceUri", "SOURCE_SHA256=$($manifest.source_zip_sha256)",
        "MANIFEST_URI=$manifestUri", "MANIFEST_SHA256=$manifestSha256",
        "AUTHORIZATION_URI=$authorizationUri", "AUTHORIZATION_SHA256=$authorizationSha256",
        "STARTUP_SHA256=$($manifest.startup_sha256)", "SCHEDULE_SHA256=$($manifest.schedule_sha256)",
        "PLAN_SHA256=$($manifest.plan_sha256)", "STATUS_SHA256=$($manifest.status_sha256)",
        "MODEL_SHA256=$($manifest.model_sha256)", "AI_PROFILES_SHA256=$($manifest.ai_profiles_sha256)",
        "PREFLIGHT_PLAN_SHA256=$($manifest.preflight_plan_sha256)",
        "PREFLIGHT_AGGREGATE_SHA256=$($manifest.preflight_aggregate_sha256)",
        "CLOSURE_SHA256=$($manifest.source_closure_sha256)",
        "SOURCE_MODEL_MANIFEST_SHA256=$($manifest.source_model_manifest_sha256)",
        "SOURCE_NATIVE_MANIFEST_SHA256=$($manifest.source_native_manifest_sha256)",
        "NATIVE_BATCH_THREADS=$NativeBatchThreads", "SYNC_INTERVAL_SECONDS=$SyncIntervalSeconds",
        "SELF_DELETE=$(if ($NoSelfDelete) { 0 } else { 1 })"
    ) -join ','
    [void](Invoke-M43A4Gcloud `
        -Arguments @(
            'compute','instances','create',$instanceName,'--project',$ProjectId,'--zone',$Zone,
            '--machine-type',$MachineType,'--provisioning-model=SPOT',
            '--instance-termination-action=DELETE','--maintenance-policy=TERMINATE',
            '--image-family=debian-12','--image-project=debian-cloud',
            '--boot-disk-size',"${BootDiskGb}GB",'--boot-disk-type',$BootDiskType,
            '--scopes=https://www.googleapis.com/auth/cloud-platform',
            '--metadata',$metadata,'--metadata-from-file',"startup-script=$startup"
        ) -TimeoutSeconds 300 -Label "create Attempt07 Spot shard $shard")
    $created.Add($shard)
}

[pscustomobject][ordered]@{
    schema = 'hu_m43_attempt07_development_spot_launch_result_v1'
    status = 'requested_bounded_spot_wave'
    run_name = $RunName
    authorization_sha256 = $authorizationSha256
    selected_shards = $selectedShards
    created_shards = @($created)
    skipped_done_shards = @($skippedDone)
    skipped_existing_instances = @($skippedExisting)
    max_wave_shards = 50
    recommended_sequence = @('0-4 operational smoke','5-49','50-99')
    machine_type = $MachineType
    native_batch_threads = 4
    fresh_root_opened_by_launcher = $false
    teacher_content_read_by_launcher = $false
    current_profile_mutated = $false
    runtime_policy_activated = $false
} | ConvertTo-Json -Depth 8
