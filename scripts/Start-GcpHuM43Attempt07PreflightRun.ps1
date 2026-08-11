param(
    [string]$ProjectId = 'ofc-solver-485418',
    [string]$Bucket = 'pokerhu-ofc-solver-485418-training',
    [string]$RunName = ('regular-hu-m43-attempt07-preflight-' + (Get-Date -Format 'yyyyMMdd-HHmmss')),
    [string]$BaseRunDir = 'outputs/gcp_runs/regular-hu-m43-attempt06-preflight-final-20260714-1154',
    [string]$PreflightPlanPath = 'configs/hu_joint_policy_m43_attempt07_preflight.json',
    [string]$Attempt07PlanPath = 'configs/hu_joint_policy_m43_attempt07.json',
    [string]$SourcePath = 'outputs/hu_joint_policy/m43_attempt06_search_quality/regular-hu-m43-attempt06-preflight-final-20260714-1154/merged/teacher.jsonl',
    [string]$RunDir,
    [string]$SpotAuthorizationPath,
    [string[]]$Jobs = @('0-4'),
    [string]$Zone = 'asia-northeast1-b',
    [string]$MachineType = 'c4-standard-4',
    [string]$BootDiskType = 'hyperdisk-balanced',
    [ValidateRange(20, 100)][int]$BootDiskGb = 50,
    [ValidateRange(1, 32)][int]$NativeBatchThreads = 4,
    [ValidateRange(10, 600)][int]$SyncIntervalSeconds = 60,
    [switch]$PackageOnly,
    [switch]$CreateInstances,
    [switch]$ResumePackage,
    [switch]$ResumeExisting,
    [switch]$SkipExistingInstances,
    [switch]$ResumeStoppedInstances,
    [switch]$NoSelfDelete
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'HuM43Attempt04Spot.Common.ps1')

$ExpectedAttempt07PlanSha256 = '8bf15f8d109a2e441e1c240e533c56df42a7e65d2eefd3b2750d1bd63758b189'
$ExpectedSourceSha256 = '6b1063589aa2ee4f3e65d9489384176a85c69abe1dfefe30ffb4474f96964903'
$ExpectedModelSha256 = 'e7fa728308c8973e2e202baf79309e30b9b6f58c37431d8ea55850390abc28c3'
$ExpectedAiProfilesSha256 = 'd2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3'
$ExpectedMachineType = 'c4-standard-4'
$JobCount = 5
$ExpectedPackageTestsPassed = 16

function Expand-M43A7JobSelection {
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
                for ($index = $first; $index -le $last; $index++) {
                    [void]$selected.Add($index)
                }
            }
            elseif ($value -match '^\d+$') { [void]$selected.Add([int]$value) }
            else { throw "Invalid Attempt07 job selection: $value" }
        }
    }
    foreach ($index in $selected) {
        if ($index -lt 0 -or $index -ge $JobCount) {
            throw "Attempt07 job index outside 0..4: $index"
        }
    }
    return @($selected)
}

function ConvertTo-M43A7VmPrefix {
    param([string]$Value)
    $result = (($Value.ToLowerInvariant() -replace '[^a-z0-9-]', '-') -replace '-+', '-').Trim('-')
    if (-not $result) { throw 'RunName cannot produce a VM prefix' }
    if ($result.Length -gt 55) { $result = $result.Substring(0, 55).TrimEnd('-') }
    return $result
}

function Invoke-M43A7Python {
    param([string[]]$Arguments, [string]$Label, [int]$TimeoutSeconds = 1800)
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

function Assert-M43A7False {
    param($Value, [string]$Label)
    if ($Value -isnot [bool] -or $Value -ne $false) { throw "$Label must be boolean false" }
}

function Test-M43A7Integer {
    param($Value)
    return ($Value -is [byte] -or $Value -is [sbyte] -or
        $Value -is [int16] -or $Value -is [uint16] -or
        $Value -is [int32] -or $Value -is [uint32] -or
        $Value -is [int64] -or $Value -is [uint64])
}

function Assert-M43A7ExactPropertySet {
    param($Value, [string[]]$Expected, [string]$Label)
    if ($null -eq $Value) { throw "$Label is missing" }
    $actualNames = @($Value.PSObject.Properties.Name | Sort-Object)
    $expectedNames = @($Expected | Sort-Object)
    if (($actualNames -join ',') -cne ($expectedNames -join ',')) {
        throw "$Label property set changed"
    }
}

function Assert-M43A7LaunchAuthorization {
    param($Authorization, [string]$ManifestSha256, $Manifest)
    Assert-M43A7ExactPropertySet -Value $Authorization -Label 'Attempt07 launch authorization' -Expected @(
        'schema', 'status', 'run_name', 'manifest_sha256', 'preflight_plan_sha256',
        'attempt07_plan_sha256', 'source_merged_sha256', 'model_sha256',
        'ai_profiles_sha256', 'machine_type', 'native_batch_threads', 'jobs',
        'source_roots', 'local_gates', 'local_evidence', 'actual_scalar_batch_result',
        'actual_operational_go_no_go', 'spot_authorized', 'new_root_generation_allowed',
        'arm_selection_allowed', 'current_profile_resolved', 'current_profile_mutated',
        'runtime_policy_activated'
    )
    if ($Authorization.schema -ne 'hu_m43_attempt07_preflight_spot_launch_authorization_v1' -or
        $Authorization.status -ne 'authorized_for_bounded_spot_preflight' -or
        [string]$Authorization.run_name -ne $RunName -or
        [string]$Authorization.manifest_sha256 -ne $ManifestSha256 -or
        [string]$Authorization.preflight_plan_sha256 -ne [string]$Manifest.preflight_plan_sha256 -or
        [string]$Authorization.attempt07_plan_sha256 -ne $ExpectedAttempt07PlanSha256 -or
        [string]$Authorization.source_merged_sha256 -ne $ExpectedSourceSha256 -or
        [string]$Authorization.model_sha256 -ne $ExpectedModelSha256 -or
        [string]$Authorization.ai_profiles_sha256 -ne $ExpectedAiProfilesSha256 -or
        [string]$Authorization.machine_type -ne $ExpectedMachineType -or
        -not (Test-M43A7Integer $Authorization.native_batch_threads) -or
        [int]$Authorization.native_batch_threads -ne 4 -or
        -not (Test-M43A7Integer $Authorization.jobs) -or
        [int]$Authorization.jobs -ne $JobCount -or
        @($Authorization.source_roots).Count -ne 3 -or
        [int]$Authorization.source_roots[0] -ne 0 -or
        [int]$Authorization.source_roots[1] -ne 1 -or
        [int]$Authorization.source_roots[2] -ne 2 -or
        $Authorization.spot_authorized -isnot [bool] -or
        $Authorization.spot_authorized -ne $true) {
        throw 'Attempt07 Spot launch authorization identity changed'
    }
    foreach ($name in @(
        'new_root_generation_allowed', 'arm_selection_allowed',
        'current_profile_resolved', 'current_profile_mutated',
        'runtime_policy_activated'
    )) {
        Assert-M43A7False -Value $Authorization.$name -Label "authorization $name"
    }
    if ([string]$Authorization.actual_scalar_batch_result -ne
            'pending_spot_preflight_receive_and_aggregate' -or
        [string]$Authorization.actual_operational_go_no_go -ne
            'pending_spot_preflight_receive_and_aggregate') {
        throw 'Attempt07 launch authorization must not pre-judge Spot results'
    }
    $expectedGateNames = @('correctness_smoke', 'determinism', 'scalar_batch_parity_test_harness', 'package_closure')
    Assert-M43A7ExactPropertySet -Value $Authorization.local_gates `
        -Expected $expectedGateNames -Label 'Attempt07 launch authorization local gates'
    foreach ($name in $expectedGateNames) {
        if ([string]$Authorization.local_gates.$name -ne 'pass') {
            throw "Attempt07 local launch gate has not passed: $name"
        }
    }
    $evidenceNames = @($Authorization.local_evidence.PSObject.Properties.Name | Sort-Object)
    if (($evidenceNames -join ',') -ne 'attempt07_pytest,package_tests,rust_parity') {
        throw 'Attempt07 launch authorization evidence set changed'
    }
    foreach ($name in $evidenceNames) {
        $row = $Authorization.local_evidence.$name
        Assert-M43A7ExactPropertySet -Value $row `
            -Expected @('receipt_sha256', 'passed', 'failed') `
            -Label "Attempt07 authorization evidence $name"
        Assert-M43A4Sha256 ([string]$row.receipt_sha256) "authorization evidence $name"
        if (-not (Test-M43A7Integer $row.passed) -or [int]$row.passed -lt 1 -or
            -not (Test-M43A7Integer $row.failed) -or [int]$row.failed -ne 0) {
            throw "Attempt07 launch authorization evidence count changed: $name"
        }
    }
    if ([int]$Authorization.local_evidence.attempt07_pytest.passed -ne 99 -or
        [int]$Authorization.local_evidence.rust_parity.passed -ne 9 -or
        [int]$Authorization.local_evidence.package_tests.passed -ne $ExpectedPackageTestsPassed) {
        throw 'Attempt07 fixed local evidence pass counts changed'
    }
}

function Publish-M43A7ImmutableObject {
    param([string]$Source, [string]$Uri)
    [void](Invoke-M43A4Gcloud `
        -Arguments @(
            'storage', 'cp', $Source, $Uri,
            '--project', $ProjectId, '--if-generation-match=0'
        ) -TimeoutSeconds 600 -Label "publish immutable $Uri")
}

function Assert-M43A7ExistingInstance {
    param($Instance, [Collections.IDictionary]$ExpectedMetadata)
    $actualMachine = ([string]$Instance.machineType -split '/')[-1]
    if ($actualMachine -ne $ExpectedMachineType -or
        [string]$Instance.scheduling.provisioningModel -ne 'SPOT') {
        throw 'existing Attempt07 instance compute shape changed'
    }
    $actualMetadata = @{}
    foreach ($item in @($Instance.metadata.items)) {
        $actualMetadata[[string]$item.key] = [string]$item.value
    }
    foreach ($key in $ExpectedMetadata.Keys) {
        if (-not $actualMetadata.ContainsKey($key) -or
            $actualMetadata[$key] -cne [string]$ExpectedMetadata[$key]) {
            throw "existing Attempt07 instance metadata changed: $key"
        }
    }
}

if ($RunName -notmatch '^[a-z0-9][a-z0-9-]{2,126}[a-z0-9]$') {
    throw 'RunName is not a safe GCP run identity'
}
if ($PackageOnly -and $CreateInstances) {
    throw 'PackageOnly and CreateInstances are mutually exclusive'
}
if (-not $PackageOnly -and -not $CreateInstances) {
    throw 'Choose PackageOnly or explicitly choose CreateInstances'
}
if ($MachineType -ne $ExpectedMachineType) {
    throw 'Attempt07 one-job-per-VM machine type is frozen at c4-standard-4'
}
if ($NativeBatchThreads -ne 4) {
    throw 'Attempt07 native batch threads are frozen at 4'
}

$script:RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $RunDir) {
    $RunDir = Join-Path (Join-Path $script:RepoRoot 'outputs/gcp_runs') $RunName
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
    throw 'Attempt07 RunDir must be exactly outputs/gcp_runs/<RunName>'
}

$resolvedBase = Resolve-M43A4Path -Path $BaseRunDir -Root $script:RepoRoot -Label 'Attempt06 base package'
$resolvedPreflightPlan = Resolve-M43A4Path -Path $PreflightPlanPath -Root $script:RepoRoot -Label 'Attempt07 preflight plan' -RequireFile
$resolvedAttempt07Plan = Resolve-M43A4Path -Path $Attempt07PlanPath -Root $script:RepoRoot -Label 'Attempt07 plan' -RequireFile
$resolvedSource = Resolve-M43A4Path -Path $SourcePath -Root $script:RepoRoot -Label 'Attempt06 merged source' -RequireFile
$startupPath = Join-Path $PSScriptRoot 'startup_hu_m43_attempt07_preflight.sh'
if (-not (Test-Path -LiteralPath $startupPath -PathType Leaf)) { throw 'Attempt07 startup worker is missing' }
if ((Get-M43A4Sha256 $resolvedAttempt07Plan) -ne $ExpectedAttempt07PlanSha256 -or
    (Get-M43A4Sha256 $resolvedSource) -ne $ExpectedSourceSha256) {
    throw 'Attempt07 frozen plan/source binding changed'
}

$packageArguments = @(
    '-B', '-m', 'ofc_regular.hu_m43_attempt07_preflight_spot', 'package',
    '--repo-root', $script:RepoRoot,
    '--run-dir', $RunDir,
    '--run-name', $RunName,
    '--base-run-dir', $resolvedBase,
    '--preflight-plan', $resolvedPreflightPlan,
    '--attempt07-plan', $resolvedAttempt07Plan,
    '--source', $resolvedSource,
    '--startup', $startupPath
)
if ($ResumePackage -or $ResumeExisting) { $packageArguments += '--resume-existing' }
$packageRaw = Invoke-M43A7Python -Arguments $packageArguments -Label 'build closed Attempt07 Spot package'
$packageLines = @($packageRaw -split "`r?`n" | Where-Object { $_.Trim() })
if ($packageLines.Count -ne 1) { throw 'Attempt07 package command emitted unexpected output' }
$packageResult = $packageLines[0] | ConvertFrom-Json
if ($packageResult.schema -ne 'hu_m43_attempt07_preflight_spot_package_result_v1' -or
    [string]$packageResult.status -notin @(
        'packaged_without_execution_or_new_root',
        'verified_existing_package_without_execution'
    ) -or [int]$packageResult.jobs -ne $JobCount) {
    throw 'Attempt07 package-only boundary changed'
}
foreach ($name in @('gcloud_invoked', 'teacher_executed', 'new_root_generated', 'current_profile_mutated')) {
    Assert-M43A7False -Value $packageResult.$name -Label "package result $name"
}

$manifestPath = Join-Path $RunDir 'manifest.json'
$schedulePath = Join-Path $RunDir 'shards_manifest.jsonl'
$sourceZipPath = Join-Path $RunDir 'source.zip'
$packagedStartupPath = Join-Path $RunDir 'startup_hu_m43_attempt07_preflight.sh'
$packagedPreflightPlanPath = Join-Path $RunDir 'hu_joint_policy_m43_attempt07_preflight.json'
$packagedAttempt07PlanPath = Join-Path $RunDir 'hu_joint_policy_m43_attempt07.json'
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$manifestSha256 = Get-M43A4Sha256 $manifestPath
Assert-M43A7ExactPropertySet -Value $manifest -Label 'Attempt07 package manifest' -Expected @(
    'schema', 'status', 'run_name', 'jobs', 'source_roots', 'machine_type',
    'native_batch_threads', 'base_attempt06_manifest_sha256',
    'base_attempt06_package_tree_sha256', 'base_attempt06_source_zip_sha256',
    'package_tree_sha256', 'overlay_closure_sha256', 'source_zip_sha256',
    'source_zip_bytes', 'startup_sha256', 'schedule_sha256', 'preflight_plan_sha256',
    'attempt07_plan_sha256', 'source_merged_sha256', 'model_sha256',
    'ai_profiles_sha256', 'new_root_generated', 'teacher_executed', 'gcloud_invoked',
    'instances_created', 'arm_selection_performed', 'current_profile_resolved',
    'current_profile_mutated', 'runtime_policy_activated'
)
if ($manifest.schema -ne 'hu_m43_attempt07_preflight_spot_package_manifest_v1' -or
    $manifest.status -ne 'packaged_attempt06_copy_plus_overlay_without_execution' -or
    [string]$manifest.run_name -ne $RunName -or
    [int]$manifest.jobs -ne $JobCount -or
    [string]$manifest.machine_type -ne $ExpectedMachineType -or
    [int]$manifest.native_batch_threads -ne 4 -or
    [string]$manifest.attempt07_plan_sha256 -ne $ExpectedAttempt07PlanSha256 -or
    [string]$manifest.source_merged_sha256 -ne $ExpectedSourceSha256 -or
    [string]$manifest.model_sha256 -ne $ExpectedModelSha256 -or
    [string]$manifest.ai_profiles_sha256 -ne $ExpectedAiProfilesSha256) {
    throw 'Attempt07 package manifest identity changed'
}
foreach ($name in @(
    'new_root_generated', 'teacher_executed', 'gcloud_invoked',
    'instances_created', 'arm_selection_performed', 'current_profile_resolved',
    'current_profile_mutated', 'runtime_policy_activated'
)) {
    Assert-M43A7False -Value $manifest.$name -Label "manifest $name"
}
foreach ($binding in @(
    @($sourceZipPath, [string]$manifest.source_zip_sha256),
    @($packagedStartupPath, [string]$manifest.startup_sha256),
    @($schedulePath, [string]$manifest.schedule_sha256),
    @($packagedPreflightPlanPath, [string]$manifest.preflight_plan_sha256),
    @($packagedAttempt07PlanPath, [string]$manifest.attempt07_plan_sha256)
)) {
    Assert-M43A4Sha256 ([string]$binding[1]) 'Attempt07 package hash'
    if ((Get-M43A4Sha256 ([string]$binding[0])) -ne [string]$binding[1]) {
        throw "Attempt07 packaged artifact changed: $($binding[0])"
    }
}

if ($PackageOnly) {
    [pscustomobject][ordered]@{
        schema = 'hu_m43_attempt07_preflight_package_only_result_v1'
        status = 'pass_no_execution_no_gcloud_no_new_root'
        run_name = $RunName
        run_dir = $RunDir
        manifest_sha256 = $manifestSha256
        preflight_plan_sha256 = [string]$manifest.preflight_plan_sha256
        jobs = $JobCount
        source_roots = @(0, 1, 2)
        machine_type = $ExpectedMachineType
        native_batch_threads = 4
        gcloud_invoked = $false
        instances_created = $false
        new_root_generated = $false
        arm_selection_performed = $false
        current_profile_resolved = $false
        current_profile_mutated = $false
    } | ConvertTo-Json -Depth 6
    exit 0
}

if (-not $SpotAuthorizationPath) {
    throw 'CreateInstances requires SpotAuthorizationPath after local gates pass'
}
$resolvedAuthorization = Resolve-M43A4Path `
    -Path $SpotAuthorizationPath -Root $script:RepoRoot `
    -Label 'Attempt07 Spot launch authorization' -RequireFile
$authorization = Get-Content -LiteralPath $resolvedAuthorization -Raw | ConvertFrom-Json
Assert-M43A7LaunchAuthorization `
    -Authorization $authorization -ManifestSha256 $manifestSha256 -Manifest $manifest
$authorizationSha256 = Get-M43A4Sha256 $resolvedAuthorization
$prefix = "gs://$Bucket/runs/$RunName"

$remoteBindings = @(
    @($sourceZipPath, "$prefix/source/ofc_regular_hu_m43_attempt07_preflight_source.zip"),
    @($packagedStartupPath, "$prefix/source/startup_hu_m43_attempt07_preflight.sh"),
    @($schedulePath, "$prefix/source/shards_manifest.jsonl"),
    @($packagedPreflightPlanPath, "$prefix/source/hu_joint_policy_m43_attempt07_preflight.json"),
    @($packagedAttempt07PlanPath, "$prefix/source/hu_joint_policy_m43_attempt07.json"),
    @($resolvedAuthorization, "$prefix/source/spot_authorization.json")
)
if ($ResumeExisting) {
    $temporaryRoot = Join-Path ([IO.Path]::GetTempPath()) ('m43a7-resume-' + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $temporaryRoot | Out-Null
    try {
        $remoteManifestUri = "$prefix/manifest.json"
        $remoteManifestPresent = Test-M43A4GcsObject `
            -Uri $remoteManifestUri -ProjectId $ProjectId
        foreach ($binding in $remoteBindings) {
            $source = [string]$binding[0]
            $uri = [string]$binding[1]
            if (Test-M43A4GcsObject -Uri $uri -ProjectId $ProjectId) {
                $destination = Join-Path $temporaryRoot ([guid]::NewGuid().ToString('N'))
                Copy-M43A4RemoteFileExact `
                    -Uri $uri -Destination $destination -ProjectId $ProjectId `
                    -ExpectedSha256 (Get-M43A4Sha256 $source)
            }
            elseif ($remoteManifestPresent) {
                throw "Attempt07 committed remote closure is missing: $uri"
            }
            else {
                Publish-M43A7ImmutableObject -Source $source -Uri $uri
            }
        }
        if ($remoteManifestPresent) {
            $destination = Join-Path $temporaryRoot 'manifest.json'
            Copy-M43A4RemoteFileExact `
                -Uri $remoteManifestUri -Destination $destination -ProjectId $ProjectId `
                -ExpectedSha256 $manifestSha256
        }
        else {
            # Complete a previously interrupted publication with the commit marker last.
            Publish-M43A7ImmutableObject -Source $manifestPath -Uri $remoteManifestUri
        }
    }
    finally { Remove-Item -LiteralPath $temporaryRoot -Recurse -Force -ErrorAction SilentlyContinue }
}
else {
    if (Test-M43A4GcsObject -Uri "$prefix/manifest.json" -ProjectId $ProjectId) {
        throw 'Attempt07 remote run already exists; use ResumeExisting for this exact closure'
    }
    foreach ($binding in $remoteBindings) {
        Publish-M43A7ImmutableObject -Source ([string]$binding[0]) -Uri ([string]$binding[1])
    }
    # The top-level manifest is the publication commit marker and is always last.
    Publish-M43A7ImmutableObject -Source $manifestPath -Uri "$prefix/manifest.json"
}

$selected = @(Expand-M43A7JobSelection -Values $Jobs)
if ($selected.Count -eq 0) { throw 'No Attempt07 jobs selected for explicit launch' }
$specs = @([IO.File]::ReadLines($schedulePath) | Where-Object { $_.Trim() } | ForEach-Object { $_ | ConvertFrom-Json })
if ($specs.Count -ne $JobCount) { throw 'Attempt07 schedule must contain exactly five jobs' }
$vmPrefix = ConvertTo-M43A7VmPrefix $RunName
$selfDelete = $(if ($NoSelfDelete) { '0' } else { '1' })
$created = @()
$started = @()
$skipped = @()
foreach ($jobIndex in $selected) {
    $spec = $specs[$jobIndex]
    if ([int]$spec.job_index -ne $jobIndex -or [int]$spec.source_root_index -notin @(0, 1, 2) -or
        [string]$spec.machine_type -ne $ExpectedMachineType -or [int]$spec.native_batch_threads -ne 4) {
        throw "Attempt07 schedule identity changed for job $jobIndex"
    }
    $vmName = "$vmPrefix-j$($jobIndex.ToString('00'))"
    $metadataMap = [ordered]@{
        RUN_NAME = $RunName
        PROJECT_ID = $ProjectId
        BUCKET = $Bucket
        JOB_INDEX = [string]$jobIndex
        SOURCE_URI = "$prefix/source/ofc_regular_hu_m43_attempt07_preflight_source.zip"
        SOURCE_SHA256 = [string]$manifest.source_zip_sha256
        STARTUP_SHA256 = [string]$manifest.startup_sha256
        MANIFEST_SHA256 = $manifestSha256
        SCHEDULE_SHA256 = [string]$manifest.schedule_sha256
        PREFLIGHT_PLAN_SHA256 = [string]$manifest.preflight_plan_sha256
        ATTEMPT07_PLAN_SHA256 = [string]$manifest.attempt07_plan_sha256
        SOURCE_MERGED_SHA256 = [string]$manifest.source_merged_sha256
        MODEL_SHA256 = [string]$manifest.model_sha256
        AI_PROFILES_SHA256 = [string]$manifest.ai_profiles_sha256
        AUTHORIZATION_SHA256 = $authorizationSha256
        OVERLAY_CLOSURE_SHA256 = [string]$manifest.overlay_closure_sha256
        PACKAGE_TREE_SHA256 = [string]$manifest.package_tree_sha256
        SYNC_INTERVAL_SECONDS = [string]$SyncIntervalSeconds
        NATIVE_BATCH_THREADS = [string]$NativeBatchThreads
        SELF_DELETE = $selfDelete
    }
    $describe = Invoke-M43A4GcloudProcess `
        -Arguments @('compute', 'instances', 'describe', $vmName, '--zone', $Zone, '--project', $ProjectId, '--format=json') `
        -TimeoutSeconds 30 -Label "describe $vmName"
    if (-not $describe.timed_out -and $describe.exit_code -eq 0) {
        $instance = ([string]$describe.stdout) | ConvertFrom-Json
        Assert-M43A7ExistingInstance -Instance $instance -ExpectedMetadata $metadataMap
        $state = [string]$instance.status
        if ($ResumeStoppedInstances) {
            if ($state -ne 'TERMINATED') {
                throw "Attempt07 instance is not resumable from state $state`: $vmName"
            }
            [void](Invoke-M43A4Gcloud `
                -Arguments @('compute', 'instances', 'start', $vmName, '--zone', $Zone, '--project', $ProjectId) `
                -TimeoutSeconds 300 -Label "resume stopped $vmName")
            $started += $vmName
            continue
        }
        if (-not $SkipExistingInstances) { throw "Attempt07 worker already exists: $vmName" }
        $skipped += $vmName
        continue
    }
    $describeMessage = ([string]$describe.stdout) + ([string]$describe.stderr)
    if ($describe.timed_out -or $describeMessage -notmatch '(?i)not found|was not found|404') {
        throw "Unable to prove Attempt07 worker absence: $vmName`: $describeMessage"
    }
    $metadata = @($metadataMap.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }) -join ','
    $create = Invoke-M43A4GcloudProcess `
        -Arguments @(
            'compute', 'instances', 'create', $vmName,
            '--project', $ProjectId, '--zone', $Zone,
            '--machine-type', $ExpectedMachineType,
            '--provisioning-model', 'SPOT', '--instance-termination-action', 'STOP',
            '--maintenance-policy', 'TERMINATE', '--no-restart-on-failure',
            '--image-family', 'debian-12', '--image-project', 'debian-cloud',
            '--boot-disk-size', "$BootDiskGb`GB", '--boot-disk-type', $BootDiskType,
            '--scopes', 'cloud-platform',
            '--labels', 'purpose=hu-m43-a07-preflight,milestone=m43-a07',
            '--metadata', $metadata,
            '--metadata-from-file', "startup-script=$packagedStartupPath"
        ) -TimeoutSeconds 300 -Label "create $vmName"
    if ($create.timed_out -or $create.exit_code -ne 0) {
        throw "Unable to submit Attempt07 worker $vmName`: $($create.stderr)$($create.stdout)"
    }
    $created += $vmName
}

[pscustomobject][ordered]@{
    schema = 'hu_m43_attempt07_preflight_spot_start_result_v1'
    status = 'submitted_explicit_bounded_preflight_jobs'
    run_name = $RunName
    manifest_sha256 = $manifestSha256
    authorization_sha256 = $authorizationSha256
    selected_jobs = $selected
    created_instances = $created
    resumed_instances = $started
    skipped_instances = $skipped
    source_roots = @(0, 1, 2)
    machine_type = $ExpectedMachineType
    native_batch_threads = 4
    result_payload_opened_by_launcher = $false
    new_root_generated = $false
    arm_selection_performed = $false
    current_profile_resolved = $false
    current_profile_mutated = $false
} | ConvertTo-Json -Depth 8
