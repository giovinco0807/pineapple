param(
    [string]$ProjectId = 'ofc-solver-485418',
    [string]$Bucket = 'pokerhu-ofc-solver-485418-training',
    [string]$RunName = ('regular-hu-m43-attempt06-top8-c8-e128-audit50-' + (Get-Date -Format 'yyyyMMdd-HHmmss')),
    [string]$PlanPath = 'configs/hu_joint_policy_m43_attempt06.json',
    [string]$StatusPath = 'configs/hu_joint_policy_m43_attempt06_status.json',
    [string]$ModelPath = 'outputs/hu_joint_policy/m43_attempt05_old_dev900_architecture_once/lambda_rank_candidate.pkl',
    [string]$RunDir,
    [string]$SpotAuthorizationPath,
    [string[]]$StartShards = @('0'),
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
    [switch]$NoSelfDelete
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'HuM43Attempt04Spot.Common.ps1')

$ExpectedPlanSha256 = '4844fb970780c04ff093eb43b1672e403f006515c47b287e6abdbea17867f5b8'
$ExpectedModelSha256 = 'e7fa728308c8973e2e202baf79309e30b9b6f58c37431d8ea55850390abc28c3'
$TotalShards = 50
$RootsPerShard = 1

if ($RunName -notmatch '^[a-z0-9][a-z0-9-]{2,126}[a-z0-9]$') {
    throw 'RunName is not a safe GCP run identity'
}
if ($PackageOnly -and $CreateInstances) {
    throw 'PackageOnly and CreateInstances are mutually exclusive'
}
if (-not $PackageOnly -and -not $CreateInstances) {
    throw 'Choose PackageOnly or explicitly choose CreateInstances'
}
if ($NativeBatchThreads -ne 4) {
    throw 'Attempt06 NativeBatchThreads is frozen at 4'
}

function Expand-M43A6ShardSelection {
    param([string[]]$Values, [int]$Count)
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
        if ($index -lt 0 -or $index -ge $Count) {
            throw "Shard index outside 0..$($Count - 1): $index"
        }
    }
    return @($selected)
}

function ConvertTo-M43A6VmPrefix {
    param([string]$Value)
    $result = (($Value.ToLowerInvariant() -replace '[^a-z0-9-]', '-') -replace '-+', '-').Trim('-')
    if (-not $result) { throw 'RunName cannot produce a VM prefix' }
    if ($result.Length -gt 54) { $result = $result.Substring(0, 54).TrimEnd('-') }
    return $result
}

function Invoke-M43A6Python {
    param([string[]]$Arguments, [string]$Label, [int]$TimeoutSeconds = 1200)
    $pythonPath = (Get-Command python -ErrorAction Stop | Select-Object -First 1).Source
    $environment = @{ PYTHONPATH = (Join-Path $script:RepoRoot 'src') }
    $result = Invoke-M43A4ProcessBounded `
        -FilePath $pythonPath -Arguments $Arguments -Label $Label `
        -TimeoutSeconds $TimeoutSeconds -Environment $environment
    if ($result.timed_out) { throw "$Label timed out" }
    if ($result.exit_code -ne 0) {
        throw "$Label failed ($($result.exit_code)): $($result.stderr)$($result.stdout)"
    }
    return [string]$result.stdout
}

function Assert-M43A6SpotAuthorization {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$PackageManifestSha256
    )
    $authorization = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    if ($authorization.schema -ne 'hu_m43_attempt06_spot_launch_authorization_v1' -or
        $authorization.status -ne 'authorized_after_local_preflight' -or
        $authorization.spot_authorized -ne $true -or
        [string]$authorization.plan_sha256 -ne $ExpectedPlanSha256 -or
        [string]$authorization.model_sha256 -ne $ExpectedModelSha256 -or
        [string]$authorization.package_manifest_sha256 -ne $PackageManifestSha256 -or
        $authorization.fresh_audit_started -ne $false -or
        $authorization.current_profile_mutated -ne $false -or
        $authorization.runtime_policy_activated -ne $false) {
        throw 'Attempt06 Spot launch authorization identity changed'
    }
    $preflight = $authorization.pre_spot_requirements
    foreach ($name in @(
        'local_correctness', 'determinism',
        'scalar_batch_exact_parity', 'latency_profile'
    )) {
        if ([string]$preflight.$name -ne 'pass') {
            throw "Attempt06 pre-Spot requirement has not passed: $name"
        }
    }
    return $authorization
}

function Publish-M43A6ImmutableObject {
    param([string]$Source, [string]$Uri)
    [void](Invoke-M43A4Gcloud `
        -Arguments @(
            'storage', 'cp', $Source, $Uri,
            '--project', $ProjectId, '--if-generation-match=0'
        ) `
        -TimeoutSeconds 600 -Label "publish immutable $Uri")
}

$script:RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $RunDir) {
    $RunDir = Join-Path (Join-Path $script:RepoRoot 'outputs/gcp_runs') $RunName
}
$RunDir = Resolve-M43A4Path -Path $RunDir -Root $script:RepoRoot -Label 'Attempt06 run directory'
Assert-M43A4UnderRoot -Path $RunDir -Root $script:RepoRoot -Label 'Attempt06 run directory'
$expectedRunDir = [IO.Path]::GetFullPath(
    (Join-Path (Join-Path $script:RepoRoot 'outputs/gcp_runs') $RunName)
).TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar)
$actualRunDir = [IO.Path]::GetFullPath($RunDir).TrimEnd(
    [IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar
)
if (-not [string]::Equals($actualRunDir, $expectedRunDir, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Attempt06 RunDir must be exactly outputs/gcp_runs/<RunName>'
}
$resolvedPlan = Resolve-M43A4Path -Path $PlanPath -Root $script:RepoRoot -Label 'Attempt06 plan' -RequireFile
$resolvedStatus = Resolve-M43A4Path -Path $StatusPath -Root $script:RepoRoot -Label 'Attempt06 status' -RequireFile
$resolvedModel = Resolve-M43A4Path -Path $ModelPath -Root $script:RepoRoot -Label 'Attempt06 ranker' -RequireFile
$startupPath = Join-Path $PSScriptRoot 'startup_hu_m43_attempt06_teacher.sh'
if (-not (Test-Path -LiteralPath $startupPath -PathType Leaf)) {
    throw 'Attempt06 startup worker is missing'
}
if ((Get-M43A4Sha256 $resolvedPlan) -ne $ExpectedPlanSha256 -or
    (Get-M43A4Sha256 $resolvedModel) -ne $ExpectedModelSha256) {
    throw 'Attempt06 frozen plan/model binding changed'
}

$packageArguments = @(
    '-B', '-m', 'ofc_regular.hu_m43_attempt06_spot', 'package',
    '--repo-root', $script:RepoRoot,
    '--run-dir', $RunDir,
    '--run-name', $RunName,
    '--plan', $resolvedPlan,
    '--status', $resolvedStatus,
    '--model', $resolvedModel,
    '--startup', $startupPath
)
# ResumePackage reuses only the immutable local closure. ResumeExisting also
# selects the remote verification branch below for an already-published run.
if ($ResumePackage -or $ResumeExisting) {
    $packageArguments += '--resume-existing'
}
$packageRaw = Invoke-M43A6Python `
    -Arguments $packageArguments -Label 'build closed Attempt06 Spot package' `
    -TimeoutSeconds 1800
$packageLines = @($packageRaw -split "`r?`n" | Where-Object { $_.Trim() })
if ($packageLines.Count -ne 1) { throw 'Attempt06 package command emitted unexpected output' }
$packageResult = $packageLines[0] | ConvertFrom-Json
if ($packageResult.schema -ne 'hu_m43_attempt06_spot_package_result_v1' -or
    $packageResult.status -ne 'packaged_without_fresh_content' -or
    $packageResult.fresh_seed_content_opened -ne $false -or
    $packageResult.teacher_executed -ne $false -or
    $packageResult.gcloud_invoked -ne $false -or
    [int]$packageResult.total_shards -ne $TotalShards -or
    [int]$packageResult.roots_per_shard -ne $RootsPerShard) {
    throw 'Attempt06 package-only boundary changed'
}

$manifestPath = Join-Path $RunDir 'manifest.json'
$schedulePath = Join-Path $RunDir 'shards_manifest.jsonl'
$sourcePath = Join-Path $RunDir 'ofc_regular_hu_m43_attempt06_teacher_source.zip'
$closurePath = Join-Path $RunDir 'source_closure_manifest.json'
$packagedStartupPath = Join-Path $RunDir 'startup_hu_m43_attempt06_teacher.sh'
$packagedPlanPath = Join-Path $RunDir 'hu_joint_policy_m43_attempt06.json'
$packagedStatusPath = Join-Path $RunDir 'hu_joint_policy_m43_attempt06_status.json'
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$manifestSha256 = Get-M43A4Sha256 $manifestPath
if ($manifest.schema -ne 'hu_m43_attempt06_spot_package_manifest_v1' -or
    $manifest.status -ne 'frozen_package_only_no_fresh_content' -or
    [string]$manifest.run_name -ne $RunName -or
    [int]$manifest.total_shards -ne $TotalShards -or
    [int]$manifest.roots_per_shard -ne $RootsPerShard -or
    [string]$manifest.plan_sha256 -ne $ExpectedPlanSha256 -or
    [string]$manifest.model_sha256 -ne $ExpectedModelSha256 -or
    $manifest.fresh_seed_content_opened -ne $false -or
    $manifest.teacher_executed -ne $false -or
    $manifest.instances_created -ne $false) {
    throw 'Attempt06 package manifest changed'
}
if ((Get-M43A4Sha256 $packagedPlanPath) -ne [string]$manifest.plan_sha256 -or
    (Get-M43A4Sha256 $packagedStatusPath) -ne [string]$manifest.status_sha256) {
    throw 'Attempt06 packaged plan/status closure changed'
}

if ($PackageOnly) {
    [pscustomobject][ordered]@{
        schema = 'hu_m43_attempt06_package_only_dry_run_v1'
        status = 'pass_no_fresh_content_no_gcloud'
        run_name = $RunName
        run_dir = $RunDir
        manifest_sha256 = $manifestSha256
        total_shards = $TotalShards
        roots_per_shard = $RootsPerShard
        fresh_seed_content_opened = $false
        teacher_executed = $false
        gcloud_invoked = $false
        instances_created = $false
        current_profile_mutated = $false
        runtime_policy_activated = $false
    } | ConvertTo-Json -Depth 5
    exit 0
}

if (-not $SpotAuthorizationPath) {
    throw 'CreateInstances requires SpotAuthorizationPath after all local preflight gates pass'
}
$resolvedAuthorization = Resolve-M43A4Path `
    -Path $SpotAuthorizationPath -Root $script:RepoRoot `
    -Label 'Attempt06 Spot authorization' -RequireFile
$authorization = Assert-M43A6SpotAuthorization `
    -Path $resolvedAuthorization -PackageManifestSha256 $manifestSha256
$authorizationSha256 = Get-M43A4Sha256 $resolvedAuthorization
$prefix = "gs://$Bucket/runs/$RunName"

if ($ResumeExisting) {
    $temporaryRoot = Join-Path ([IO.Path]::GetTempPath()) ('m43a6-resume-' + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $temporaryRoot | Out-Null
    try {
        foreach ($binding in @(
            @("$prefix/manifest.json", $manifestPath),
            @("$prefix/source/ofc_regular_hu_m43_attempt06_teacher_source.zip", $sourcePath),
            @("$prefix/source/startup_hu_m43_attempt06_teacher.sh", $packagedStartupPath),
            @("$prefix/source/shards_manifest.jsonl", $schedulePath),
            @("$prefix/source/source_closure_manifest.json", $closurePath),
            @("$prefix/source/hu_joint_policy_m43_attempt06.json", $packagedPlanPath),
            @("$prefix/source/hu_joint_policy_m43_attempt06_status.json", $packagedStatusPath),
            @("$prefix/source/spot_authorization.json", $resolvedAuthorization)
        )) {
            $destination = Join-Path $temporaryRoot ([IO.Path]::GetFileName([string]$binding[1]))
            Copy-M43A4RemoteFileExact `
                -Uri ([string]$binding[0]) -Destination $destination -ProjectId $ProjectId `
                -ExpectedSha256 (Get-M43A4Sha256 ([string]$binding[1]))
        }
    }
    finally {
        Remove-Item -LiteralPath $temporaryRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}
else {
    if (Test-M43A4GcsObject -Uri "$prefix/manifest.json" -ProjectId $ProjectId) {
        throw 'Attempt06 remote run already exists; use ResumeExisting only for the same closure'
    }
    foreach ($binding in @(
        @($manifestPath, "$prefix/manifest.json"),
        @($sourcePath, "$prefix/source/ofc_regular_hu_m43_attempt06_teacher_source.zip"),
        @($packagedStartupPath, "$prefix/source/startup_hu_m43_attempt06_teacher.sh"),
        @($schedulePath, "$prefix/source/shards_manifest.jsonl"),
        @($closurePath, "$prefix/source/source_closure_manifest.json"),
        @($packagedPlanPath, "$prefix/source/hu_joint_policy_m43_attempt06.json"),
        @($packagedStatusPath, "$prefix/source/hu_joint_policy_m43_attempt06_status.json"),
        @($resolvedAuthorization, "$prefix/source/spot_authorization.json")
    )) {
        Publish-M43A6ImmutableObject -Source ([string]$binding[0]) -Uri ([string]$binding[1])
    }
}

$selected = @(Expand-M43A6ShardSelection -Values $StartShards -Count $TotalShards)
if ($selected.Count -eq 0) { throw 'No Attempt06 shards selected for explicit launch' }
$vmPrefix = ConvertTo-M43A6VmPrefix $RunName
$selfDelete = $(if ($NoSelfDelete) { '0' } else { '1' })
$created = @()
$skipped = @()
foreach ($shard in $selected) {
    $vmName = "$vmPrefix-s$($shard.ToString('000'))"
    $describe = Invoke-M43A4GcloudProcess `
        -Arguments @('compute', 'instances', 'describe', $vmName, '--zone', $Zone, '--project', $ProjectId, '--format=json') `
        -TimeoutSeconds 30 -Label "describe $vmName"
    if (-not $describe.timed_out -and $describe.exit_code -eq 0) {
        if (-not $SkipExistingInstances) { throw "Attempt06 worker already exists: $vmName" }
        $skipped += $vmName
        continue
    }
    $describeMessage = ([string]$describe.stdout) + ([string]$describe.stderr)
    if ($describe.timed_out -or $describeMessage -notmatch '(?i)not found|was not found|404') {
        throw "Unable to prove Attempt06 worker absence: $vmName`: $describeMessage"
    }
    $metadata = @(
        "RUN_NAME=$RunName", "PROJECT_ID=$ProjectId", "BUCKET=$Bucket", "SHARD_INDEX=$shard",
        "SOURCE_URI=$prefix/source/ofc_regular_hu_m43_attempt06_teacher_source.zip",
        "SOURCE_SHA256=$($manifest.source_zip_sha256)", "STARTUP_SHA256=$($manifest.startup_sha256)",
        "MANIFEST_SHA256=$manifestSha256", "SCHEDULE_SHA256=$($manifest.schedule_sha256)",
        "PLAN_SHA256=$($manifest.plan_sha256)", "STATUS_SHA256=$($manifest.status_sha256)",
        "MODEL_SHA256=$($manifest.model_sha256)", "CLOSURE_SHA256=$($manifest.source_closure_sha256)",
        "SOURCE_MODEL_MANIFEST_SHA256=$($manifest.source_model_manifest_sha256)",
        "SOURCE_NATIVE_MANIFEST_SHA256=$($manifest.source_native_manifest_sha256)",
        "AUTHORIZATION_SHA256=$authorizationSha256", "SYNC_INTERVAL_SECONDS=$SyncIntervalSeconds",
        "NATIVE_BATCH_THREADS=$NativeBatchThreads", "SELF_DELETE=$selfDelete"
    ) -join ','
    $create = Invoke-M43A4GcloudProcess `
        -Arguments @(
            'compute', 'instances', 'create', $vmName,
            '--project', $ProjectId, '--zone', $Zone,
            '--machine-type', $MachineType,
            '--provisioning-model', 'SPOT', '--instance-termination-action', 'STOP',
            '--maintenance-policy', 'TERMINATE', '--no-restart-on-failure',
            '--image-family', 'debian-12', '--image-project', 'debian-cloud',
            '--boot-disk-size', "$BootDiskGb`GB", '--boot-disk-type', $BootDiskType,
            '--scopes', 'cloud-platform',
            '--labels', 'purpose=hu-m43-a06-teacher,milestone=m43-a06',
            '--metadata', $metadata,
            '--metadata-from-file', "startup-script=$packagedStartupPath"
        ) `
        -TimeoutSeconds 300 -Label "create $vmName"
    if ($create.timed_out -or $create.exit_code -ne 0) {
        throw "Unable to submit Attempt06 worker $vmName`: $($create.stderr)$($create.stdout)"
    }
    $created += $vmName
}

[pscustomobject][ordered]@{
    schema = 'hu_m43_attempt06_spot_start_result_v1'
    status = 'submitted_explicit_one_root_shards'
    run_name = $RunName
    manifest_sha256 = $manifestSha256
    authorization_sha256 = $authorizationSha256
    selected_shards = $selected
    created_instances = $created
    skipped_instances = $skipped
    roots_per_shard = $RootsPerShard
    fresh_content_opened_by_launcher = $false
    worker_claims_before_root_materialization = $true
    current_profile_mutated = $false
    runtime_policy_activated = $false
} | ConvertTo-Json -Depth 8
