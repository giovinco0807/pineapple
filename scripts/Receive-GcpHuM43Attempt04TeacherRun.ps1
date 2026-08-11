param(
    [string]$ProjectId = 'ofc-solver-485418',
    [string]$Bucket = 'pokerhu-ofc-solver-485418-training',
    [Parameter(Mandatory = $true)][string]$RunName,
    [Parameter(Mandatory = $true)]
    [ValidateSet(
        'precal_holdout',
        'calibration_safety_fit',
        'calibration_threshold_lock',
        'locked_holdout'
    )][string]$Role,
    [Parameter(Mandatory = $true)][string]$RoleOpenAuthorizationPath,
    [Parameter(Mandatory = $true)][string]$RoleConsumptionMarkerPath,
    [string]$PlanPath = 'configs/hu_joint_policy_m43_attempt04.json',
    [string]$RunDir,
    [string]$OutputDir,
    [string]$TrainingModule = 'ofc_regular.hu_m43_attempt04_training',
    [ValidateRange(1, 8)][int]$MaxParallel = 8,
    [switch]$ResumeClaim,
    [ValidateSet('none', 'after_publish', 'after_reaudit')]
    [string]$TestOnlyFailureInjection = 'none'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'HuM43Attempt04Spot.Common.ps1')

if ($RunName -notmatch '^[a-z0-9][a-z0-9-]{2,126}[a-z0-9]$') {
    throw 'RunName is not a safe GCP run identity'
}
if ($TrainingModule -notmatch '^[A-Za-z_][A-Za-z0-9_.]*$') {
    throw 'TrainingModule is not a safe Python module name'
}

function Invoke-M43A4TrainingCli {
    param(
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$Label,
        [ValidateRange(1, 3600)][int]$TimeoutSeconds = 300
    )
    $result = Invoke-M43A4ProcessBounded `
        -FilePath $script:PythonPath `
        -Arguments (@('-B', '-m', $TrainingModule) + $Arguments) `
        -TimeoutSeconds $TimeoutSeconds `
        -Label $Label `
        -Environment @{
            PYTHONPATH = (Join-Path $script:RepoRoot 'src')
            PYTHONHASHSEED = '0'
        }
    if ($result.timed_out -or $result.exit_code -ne 0) {
        throw "$Label failed: $($result.stderr)$($result.stdout)"
    }
}

function Sync-M43A4TreeBestEffort {
    param([Parameter(Mandatory = $true)][string]$Path)
    $script = @'
import os,pathlib,sys
root=pathlib.Path(sys.argv[1]).resolve()
for item in sorted(value for value in root.rglob("*") if value.is_file()):
    fd=os.open(str(item),os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
for directory in (root,root.parent):
    try:
        fd=os.open(str(directory),os.O_RDONLY)
    except OSError:
        continue
    try:
        try: os.fsync(fd)
        except OSError: pass
    finally: os.close(fd)
'@
    $result = Invoke-M43A4ProcessBounded `
        -FilePath $script:PythonPath `
        -Arguments @('-B', '-c', $script, $Path) `
        -TimeoutSeconds 300 `
        -Label 'Attempt04 role durability sync'
    if ($result.timed_out -or $result.exit_code -ne 0) {
        throw "Attempt04 role durability sync failed: $($result.stderr)$($result.stdout)"
    }
}

function Publish-M43A4DirectoryAtomic {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination
    )
    $sourceFull = [IO.Path]::GetFullPath($Source)
    $destinationFull = [IO.Path]::GetFullPath($Destination)
    if (-not (Test-Path -LiteralPath $sourceFull -PathType Container)) {
        throw "Atomic publication source is missing: $sourceFull"
    }
    if (Test-Path -LiteralPath $destinationFull) {
        throw "Atomic publication destination already exists: $destinationFull"
    }
    [IO.Directory]::Move($sourceFull, $destinationFull)
}

function Read-M43A4ExistingClaim {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$RoleIdentitySha256,
        [Parameter(Mandatory = $true)][string]$AuthorizationFileSha256,
        [Parameter(Mandatory = $true)][string]$PlanFileSha256,
        [Parameter(Mandatory = $true)][string]$ManifestFileSha256,
        [Parameter(Mandatory = $true)][string]$ScheduleFileSha256
    )
    $claim = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    if ($claim.schema -ne 'hu_m43_attempt04_role_consumption_v1' -or
        $claim.status -ne 'consumed_before_any_role_stat_hash_download_or_read' -or
        [string]$claim.logical_role -ne $Role -or
        [string]$claim.role_identity_sha256 -ne $RoleIdentitySha256 -or
        [string]$claim.run_name -ne $RunName -or
        [string]$claim.authorization_file_sha256 -ne $AuthorizationFileSha256 -or
        [string]$claim.plan_file_sha256 -ne $PlanFileSha256 -or
        [string]$claim.manifest_file_sha256 -ne $ManifestFileSha256 -or
        [string]$claim.schedule_file_sha256 -ne $ScheduleFileSha256 -or
        [bool]$claim.result_objects_addressed_when_claimed -or
        [bool]$claim.other_role_result_objects_addressed -or
        [bool]$claim.current_profile_mutated -or
        [bool]$claim.runtime_policy_activated) {
        throw 'Attempt04 existing role claim does not match the frozen role closure'
    }
    return $claim
}

function Assert-M43A4TeacherManifest {
    param(
        [Parameter(Mandatory = $true)]$Manifest,
        [Parameter(Mandatory = $true)][string]$ManifestPath,
        [Parameter(Mandatory = $true)][string]$SchedulePath
    )
    if ($Manifest.schema -ne 'hu_m43_attempt04_teacher_spot_manifest_v1' -or
        $Manifest.status -ne 'frozen' -or
        [string]$Manifest.run_name -ne $RunName -or
        [string]$Manifest.project_id -ne $ProjectId -or
        [string]$Manifest.bucket -ne $Bucket -or
        [int]$Manifest.total_roots -ne 700 -or
        [int]$Manifest.total_shards -ne 70 -or
        [int]$Manifest.roots_per_shard -ne 10 -or
        [int]$Manifest.candidate_samples -ne 2 -or
        [int]$Manifest.evaluation_samples -ne 128 -or
        [string]$Manifest.schedule_sha256 -ne (Get-M43A4Sha256 $SchedulePath) -or
        [bool]$Manifest.current_profile_mutated -or
        [bool]$Manifest.runtime_policy_activated -or
        [bool]$Manifest.full_replacement) {
        throw 'Attempt04 teacher manifest changed'
    }
    Assert-M43A4Sha256 (Get-M43A4Sha256 $ManifestPath) 'manifest hash'
    foreach ($value in @(
        $Manifest.plan_file_sha256,
        $Manifest.model_freeze_file_sha256,
        $Manifest.training_freeze_file_sha256,
        $Manifest.teacher_contract_file_sha256,
        $Manifest.preflight_file_sha256,
        $Manifest.schedule_sha256,
        $Manifest.source_sha256,
        $Manifest.startup_sha256,
        $Manifest.model_manifest_sha256,
        $Manifest.native_manifest_sha256
    )) {
        Assert-M43A4Sha256 $value 'teacher closure hash'
    }
}

function Assert-M43A4RoleSchedule {
    param(
        [Parameter(Mandatory = $true)][object[]]$AllSpecs,
        [Parameter(Mandatory = $true)][object[]]$RoleSpecs
    )
    $expectedCounts = [ordered]@{
        precal_holdout = 30
        calibration_safety_fit = 10
        calibration_threshold_lock = 10
        locked_holdout = 20
    }
    $expectedStarts = [ordered]@{
        precal_holdout = 0
        calibration_safety_fit = 30
        calibration_threshold_lock = 40
        locked_holdout = 50
    }
    if ($AllSpecs.Count -ne 70 -or
        $RoleSpecs.Count -ne [int]$expectedCounts[$Role]) {
        throw "Attempt04 schedule role count changed: $Role"
    }
    $prefixes = [Collections.Generic.HashSet[string]]::new(
        [StringComparer]::Ordinal
    )
    for ($roleShard = 0; $roleShard -lt $RoleSpecs.Count; $roleShard++) {
        $spec = $RoleSpecs[$roleShard]
        $expectedShard = [int]$expectedStarts[$Role] + $roleShard
        if ($spec.schema -ne 'hu_m43_attempt04_teacher_shard_v1' -or
            [string]$spec.logical_role -ne $Role -or
            [string]$spec.split -ne 'train' -or
            [int]$spec.shard -ne $expectedShard -or
            [int]$spec.role_shard -ne $roleShard -or
            [int]$spec.roots -ne 10 -or
            [int]$spec.candidate_samples -ne 2 -or
            [int]$spec.evaluation_samples -ne 128 -or
            -not $prefixes.Add([string]$spec.output_prefix)) {
            throw "Attempt04 role schedule changed at $Role shard $roleShard"
        }
    }
}

function Complete-M43A4PublishedRoleLocal {
    param(
        [Parameter(Mandatory = $true)][string]$OutputDirectory,
        [Parameter(Mandatory = $true)][string]$ClaimFileSha256,
        [Parameter(Mandatory = $true)][bool]$RecoveredExistingOutput
    )
    $closureBindings = @(
        @('closure/manifest.json', $script:ManifestFileSha256),
        @('closure/shards_manifest.jsonl', $script:ScheduleFileSha256),
        @('closure/teacher_role_manifest.json', $script:SelectedRoleManifest.file_sha256),
        @('closure/teacher_contract.json', $script:Manifest.teacher_contract_file_sha256),
        @('closure/teacher_preflight_receipt.json', $script:Manifest.preflight_file_sha256),
        @('closure/teacher_source.zip', $script:Manifest.source_sha256),
        @('closure/startup_teacher.sh', $script:Manifest.startup_sha256),
        @('closure/source_model_manifest.json', $script:Manifest.model_manifest_sha256),
        @('closure/source_native_manifest.json', $script:Manifest.native_manifest_sha256)
    )
    foreach ($binding in $closureBindings) {
        $path = Join-Path $OutputDirectory ([string]$binding[0])
        if (-not (Test-Path -LiteralPath $path -PathType Leaf) -or
            (Get-M43A4Sha256 $path) -ne [string]$binding[1]) {
            throw "Attempt04 published local closure changed: $($binding[0])"
        }
    }
    $publishedTeacher = Join-Path $OutputDirectory 'merged/teacher.jsonl'
    $publishedReceipt = Join-Path $OutputDirectory 'merged/receive_receipt.json'
    if (-not (Test-Path -LiteralPath $publishedTeacher -PathType Leaf) -or
        -not (Test-Path -LiteralPath $publishedReceipt -PathType Leaf)) {
        throw 'Attempt04 published role is missing its teacher or receipt'
    }
    $receipt = Get-Content -LiteralPath $publishedReceipt -Raw | ConvertFrom-Json
    if ($receipt.schema -ne 'hu_m43_attempt04_teacher_role_receive_receipt_v1' -or
        $receipt.status -ne 'verified_role_after_frozen_claim' -or
        [string]$receipt.logical_role -ne $Role -or
        [string]$receipt.role_identity_sha256 -ne $script:RoleIdentitySha256 -or
        [string]$receipt.consumption_marker_file_sha256 -ne $ClaimFileSha256) {
        throw 'Attempt04 published role receipt no longer matches its exact claim'
    }
    Assert-M43A4Sha256 ([string]$receipt.receipt_sha256) 'published receipt self hash'

    $finalizeRoot = Join-Path $script:RunDir 'local_finalize_staging'
    New-Item -ItemType Directory -Force -Path $finalizeRoot | Out-Null
    $finalizeStage = Join-Path $finalizeRoot (
        $Role + '-' + [guid]::NewGuid().ToString('N')
    )
    New-Item -ItemType Directory -Path $finalizeStage | Out-Null
    $newAuditPath = Join-Path $finalizeStage 'published_role_audit.json'
    try {
        Invoke-M43A4TrainingCli `
            -Arguments @(
                'validate-published-teacher-role',
                '--plan', $script:PlanPath,
                '--repo-root', $script:RepoRoot,
                '--role', $Role,
                '--output-dir', $OutputDirectory,
                '--role-manifest', (Join-Path $OutputDirectory 'closure/teacher_role_manifest.json'),
                '--authorization', $script:RoleOpenAuthorizationPath,
                '--consumption-marker', $script:RoleConsumptionMarkerPath,
                '--output', $newAuditPath
            ) `
            -Label "local-only re-audit published Attempt04 $Role teacher role" `
            -TimeoutSeconds 900
        $newAudit = Get-Content -LiteralPath $newAuditPath -Raw | ConvertFrom-Json
        if ($newAudit.schema -ne 'hu_m43_attempt04_published_teacher_role_audit_v1' -or
            $newAudit.status -ne 'pass' -or
            [string]$newAudit.logical_role -ne $Role -or
            [string]$newAudit.role_identity_sha256 -ne $script:RoleIdentitySha256) {
            throw 'Attempt04 published role local re-audit did not pass'
        }
        $canonicalAuditPath = $OutputDirectory + '.published_role_audit.json'
        if (Test-Path -LiteralPath $canonicalAuditPath -PathType Leaf) {
            if ((Get-M43A4Sha256 $canonicalAuditPath) -ne (Get-M43A4Sha256 $newAuditPath)) {
                throw 'Attempt04 canonical published audit differs from fresh local re-audit'
            }
        }
        else {
            [IO.File]::Move($newAuditPath, $canonicalAuditPath)
        }
        if ($TestOnlyFailureInjection -eq 'after_reaudit') {
            throw 'TEST_ONLY_ATTEMPT04_FAILURE_AFTER_REAUDIT'
        }
        $auditFileSha256 = Get-M43A4Sha256 $canonicalAuditPath
        $receiptFileSha256 = Get-M43A4Sha256 $publishedReceipt
        $finalizationPath = $OutputDirectory + '.finalized.json'
        $finalization = [ordered]@{
            schema = 'hu_m43_attempt04_teacher_role_finalization_v1'
            status = 'local_final_path_reaudit_pass'
            run_name = $RunName
            logical_role = $Role
            role_identity_sha256 = $script:RoleIdentitySha256
            authorization_file_sha256 = $script:AuthorizationFileSha256
            claim_file_sha256 = $ClaimFileSha256
            plan_file_sha256 = $script:PlanFileSha256
            manifest_file_sha256 = $script:ManifestFileSha256
            schedule_file_sha256 = $script:ScheduleFileSha256
            receipt_file_sha256 = $receiptFileSha256
            published_audit_file_sha256 = $auditFileSha256
            cloud_result_objects_addressed_during_finalization = $false
            other_role_result_objects_addressed = $false
            current_profile_mutated = $false
            runtime_policy_activated = $false
        }
        if (Test-Path -LiteralPath $finalizationPath -PathType Leaf) {
            $existingFinalization = Get-Content `
                -LiteralPath $finalizationPath -Raw | ConvertFrom-Json
            foreach ($key in $finalization.Keys) {
                if ([string]$existingFinalization.$key -ne [string]$finalization[$key]) {
                    throw "Attempt04 existing local finalization changed: $key"
                }
            }
        }
        else {
            Write-M43A4Utf8CreateNew `
                -Path $finalizationPath `
                -Text (($finalization | ConvertTo-Json -Depth 12) + "`n")
        }
        return [pscustomobject][ordered]@{
            published_receipt = $publishedReceipt
            receipt_file_sha256 = $receiptFileSha256
            published_audit = $canonicalAuditPath
            published_audit_file_sha256 = $auditFileSha256
            finalization = $finalizationPath
            finalization_file_sha256 = Get-M43A4Sha256 $finalizationPath
            recovered_existing_output = $RecoveredExistingOutput
        }
    }
    finally {
        Remove-Item -LiteralPath $finalizeStage -Recurse -Force -ErrorAction SilentlyContinue
    }
}

function Invoke-M43A4PostPublishFinalization {
    param(
        [Parameter(Mandatory = $true)][string]$OutputDirectory,
        [Parameter(Mandatory = $true)][string]$ClaimFileSha256,
        [Parameter(Mandatory = $true)][bool]$RecoveredExistingOutput
    )
    if (-not $RecoveredExistingOutput -and
        $TestOnlyFailureInjection -eq 'after_publish') {
        throw 'TEST_ONLY_ATTEMPT04_FAILURE_AFTER_PUBLISH'
    }
    return Complete-M43A4PublishedRoleLocal `
        -OutputDirectory $OutputDirectory `
        -ClaimFileSha256 $ClaimFileSha256 `
        -RecoveredExistingOutput $RecoveredExistingOutput
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$PythonPath = [string](Get-Command python -ErrorAction Stop | Select-Object -First 1).Source
$PlanPath = Resolve-M43A4Path `
    -Path $PlanPath -Root $RepoRoot -Label 'Attempt04 plan' -RequireFile
$RoleOpenAuthorizationPath = Resolve-M43A4Path `
    -Path $RoleOpenAuthorizationPath -Root $RepoRoot `
    -Label 'Attempt04 role open authorization' -RequireFile
if (-not $RunDir) {
    $RunDir = Join-Path (Join-Path $RepoRoot 'outputs/gcp_runs') $RunName
}
$RunDir = Resolve-M43A4Path `
    -Path $RunDir -Root $RepoRoot -Label 'Attempt04 run directory'
Assert-M43A4UnderRoot -Path $RunDir -Root $RepoRoot -Label 'Attempt04 run directory'
if (-not $OutputDir) {
    $OutputDir = Join-Path (
        Join-Path $RepoRoot 'outputs/hu_joint_policy/m43_attempt04_teacher_roles'
    ) $Role
}
$OutputDir = Resolve-M43A4Path `
    -Path $OutputDir -Root $RepoRoot -Label 'Attempt04 role output'
Assert-M43A4UnderRoot -Path $OutputDir -Root $RepoRoot -Label 'Attempt04 role output'
$outputExists = Test-Path -LiteralPath $OutputDir
if ($outputExists -and -not (Test-Path -LiteralPath $OutputDir -PathType Container)) {
    throw "Attempt04 role output path is not a directory: $OutputDir"
}

$manifestPath = Join-Path $RunDir 'manifest.json'
$schedulePath = Join-Path $RunDir 'shards_manifest.jsonl'
foreach ($path in @($manifestPath, $schedulePath)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Attempt04 immutable run input is missing: $path"
    }
}
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
Assert-M43A4TeacherManifest `
    -Manifest $manifest -ManifestPath $manifestPath -SchedulePath $schedulePath
if ((Get-M43A4Sha256 $PlanPath) -ne [string]$manifest.plan_file_sha256) {
    throw 'Attempt04 plan does not match the teacher manifest'
}
$expectedRoles = @(
    'precal_holdout',
    'calibration_safety_fit',
    'calibration_threshold_lock',
    'locked_holdout'
)
$manifestRoleNames = @($manifest.role_manifests.PSObject.Properties.Name)
if ((@($manifestRoleNames | Sort-Object) -join '|') -ne
    (@($expectedRoles | Sort-Object) -join '|')) {
    throw 'Attempt04 manifest role set changed'
}
$selectedRoleManifest = $manifest.role_manifests.PSObject.Properties[$Role].Value
Assert-M43A4Sha256 ([string]$selectedRoleManifest.file_sha256) 'role manifest file hash'
Assert-M43A4Sha256 ([string]$selectedRoleManifest.role_identity_sha256) 'role identity hash'
if ([string]$selectedRoleManifest.gcs_path -ne
    ('source/teacher_role_manifests/' + $Role + '.json')) {
    throw 'Attempt04 selected role manifest GCS path changed'
}

$allSpecs = @(
    [IO.File]::ReadLines($schedulePath) |
        Where-Object { $_.Trim() } |
        ForEach-Object { $_ | ConvertFrom-Json }
)
# Reduce to exactly one role before a result object name can be formed.
$roleSpecs = @($allSpecs | Where-Object { [string]$_.logical_role -eq $Role })
Assert-M43A4RoleSchedule -AllSpecs $allSpecs -RoleSpecs $roleSpecs

$authorization = Get-Content `
    -LiteralPath $RoleOpenAuthorizationPath -Raw | ConvertFrom-Json
if ($authorization.schema -ne 'hu_m43_attempt04_role_open_authorization_v1' -or
    $authorization.status -ne 'authorized_to_create_role_open_claim' -or
    [string]$authorization.logical_role -ne $Role) {
    throw 'Attempt04 role open authorization identity changed'
}
$roleIdentitySha256 = [string]$authorization.role_identity_sha256
Assert-M43A4Sha256 $roleIdentitySha256 'role identity hash'
if ($roleIdentitySha256 -ne [string]$selectedRoleManifest.role_identity_sha256) {
    throw 'Attempt04 authorization role identity does not match the run manifest'
}

$RoleConsumptionMarkerPath = Resolve-M43A4Path `
    -Path $RoleConsumptionMarkerPath -Root $RepoRoot `
    -Label 'Attempt04 role consumption marker'
$expectedMarkerPath = [IO.Path]::GetFullPath((Join-Path (
    Join-Path $RepoRoot (
        'outputs/hu_joint_policy/m43_attempt04_role_consumption/' +
        $roleIdentitySha256
    )
) 'M43_ATTEMPT04_ROLE_CONSUMED.json'))
if (-not $RoleConsumptionMarkerPath.Equals(
        $expectedMarkerPath,
        [StringComparison]::OrdinalIgnoreCase
    )) {
    throw "RoleConsumptionMarkerPath must be canonical: $expectedMarkerPath"
}
$claimExists = Test-Path -LiteralPath $RoleConsumptionMarkerPath -PathType Leaf
$authorizationFileSha256 = Get-M43A4Sha256 $RoleOpenAuthorizationPath
$planFileSha256 = Get-M43A4Sha256 $PlanPath
$manifestFileSha256 = Get-M43A4Sha256 $manifestPath
$scheduleFileSha256 = Get-M43A4Sha256 $schedulePath
$existingClaim = $null
if ($ResumeClaim) {
    if (-not $claimExists) {
        throw 'ResumeClaim requires the exact existing canonical role claim'
    }
    $existingClaim = Read-M43A4ExistingClaim `
        -Path $RoleConsumptionMarkerPath `
        -RoleIdentitySha256 $roleIdentitySha256 `
        -AuthorizationFileSha256 $authorizationFileSha256 `
        -PlanFileSha256 $planFileSha256 `
        -ManifestFileSha256 $manifestFileSha256 `
        -ScheduleFileSha256 $scheduleFileSha256
}
elseif ($claimExists) {
    throw 'Attempt04 role was already claimed; use ResumeClaim for this exact role only'
}

if ($outputExists) {
    if (-not $ResumeClaim) {
        throw "Attempt04 role output already exists: $OutputDir"
    }
    $claimFileSha256 = Get-M43A4Sha256 $RoleConsumptionMarkerPath
    $localFinalization = Invoke-M43A4PostPublishFinalization `
        -OutputDirectory $OutputDir `
        -ClaimFileSha256 $claimFileSha256 `
        -RecoveredExistingOutput $true
    [pscustomobject][ordered]@{
        schema = 'hu_m43_attempt04_teacher_role_receive_result_v1'
        status = 'recovered_local_finalization_without_cloud_access'
        run_name = $RunName
        logical_role = $Role
        role_identity_sha256 = $roleIdentitySha256
        shards = $roleSpecs.Count
        roots = $roleSpecs.Count * 10
        output_dir = $OutputDir
        merged_teacher = Join-Path $OutputDir 'merged/teacher.jsonl'
        receipt = [string]$localFinalization.published_receipt
        receipt_file_sha256 = [string]$localFinalization.receipt_file_sha256
        published_audit = [string]$localFinalization.published_audit
        published_audit_file_sha256 = [string]$localFinalization.published_audit_file_sha256
        finalization = [string]$localFinalization.finalization
        finalization_file_sha256 = [string]$localFinalization.finalization_file_sha256
        consumption_marker = $RoleConsumptionMarkerPath
        consumption_marker_file_sha256 = $claimFileSha256
        resumed_existing_claim = $true
        cloud_result_objects_addressed_during_recovery = $false
        other_role_result_objects_addressed = $false
        current_profile_mutated = $false
        runtime_policy_activated = $false
    } | ConvertTo-Json -Depth 10
    exit 0
}

$outputParent = Split-Path -Parent $OutputDir
New-Item -ItemType Directory -Force -Path $outputParent | Out-Null
$stagingRoot = Join-Path $outputParent (
    '.m43a4-' + $Role + '-' + [guid]::NewGuid().ToString('N')
)
New-Item -ItemType Directory -Path $stagingRoot | Out-Null
$closureDir = Join-Path $stagingRoot 'closure'
$shardsDir = Join-Path $stagingRoot 'shards'
$mergedDir = Join-Path $stagingRoot 'merged'
$auditsDir = Join-Path $stagingRoot 'audits'
New-Item -ItemType Directory -Path $closureDir, $shardsDir, $mergedDir, $auditsDir | Out-Null

$prefix = "gs://$Bucket/runs/$RunName"
$closureCopies = @(
    @('manifest.json', "$prefix/manifest.json", (Get-M43A4Sha256 $manifestPath)),
    @('shards_manifest.jsonl', "$prefix/source/shards_manifest.jsonl", $manifest.schedule_sha256),
    @('teacher_contract.json', "$prefix/source/teacher_contract.json", $manifest.teacher_contract_file_sha256),
    @('teacher_preflight_receipt.json', "$prefix/source/teacher_preflight_receipt.json", $manifest.preflight_file_sha256),
    @('teacher_source.zip', "$prefix/source/ofc_regular_hu_m43_attempt04_teacher_source.zip", $manifest.source_sha256),
    @('startup_teacher.sh', "$prefix/source/startup_hu_m43_attempt04_teacher.sh", $manifest.startup_sha256),
    @('source_model_manifest.json', "$prefix/source/source_model_manifest.json", $manifest.model_manifest_sha256),
    @('source_native_manifest.json', "$prefix/source/source_native_manifest.json", $manifest.native_manifest_sha256),
    @(
        'teacher_role_manifest.json',
        "$prefix/source/teacher_role_manifests/$Role.json",
        $selectedRoleManifest.file_sha256
    )
)
foreach ($copy in $closureCopies) {
    Copy-M43A4RemoteFileExact `
        -Uri ([string]$copy[1]) `
        -Destination (Join-Path $closureDir ([string]$copy[0])) `
        -ProjectId $ProjectId `
        -ExpectedSha256 ([string]$copy[2])
}

$authorizationAuditPath = Join-Path $auditsDir 'authorization_audit.json'
Invoke-M43A4TrainingCli `
    -Arguments @(
        'validate-open-authorization',
        '--plan', $PlanPath,
        '--repo-root', $RepoRoot,
        '--role', $Role,
        '--authorization', $RoleOpenAuthorizationPath,
        '--manifest', (Join-Path $closureDir 'manifest.json'),
        '--schedule', (Join-Path $closureDir 'shards_manifest.jsonl'),
        '--role-manifest', (Join-Path $closureDir 'teacher_role_manifest.json'),
        '--output', $authorizationAuditPath
    ) `
    -Label "validate Attempt04 $Role authorization"
if (-not (Test-Path -LiteralPath $authorizationAuditPath -PathType Leaf)) {
    throw 'Attempt04 authorization validator produced no audit'
}
$authorizationAudit = Get-Content `
    -LiteralPath $authorizationAuditPath -Raw | ConvertFrom-Json
if ($authorizationAudit.schema -ne 'hu_m43_attempt04_role_open_authorization_audit_v1' -or
    $authorizationAudit.status -ne 'pass' -or
    [string]$authorizationAudit.logical_role -ne $Role -or
    [string]$authorizationAudit.role_identity_sha256 -ne $roleIdentitySha256) {
    throw 'Attempt04 role authorization audit did not pass'
}

# A resume reuses this exact immutable claim and cannot select another role.
if (-not $ResumeClaim) {
    # This immutable CreateNew claim is intentionally before the first result
    # object is named, described, downloaded, hashed, or decoded.
    $claim = [ordered]@{
        schema = 'hu_m43_attempt04_role_consumption_v1'
        status = 'consumed_before_any_role_stat_hash_download_or_read'
        logical_role = $Role
        role_identity_sha256 = $roleIdentitySha256
        run_name = $RunName
        authorization_file_sha256 = $authorizationFileSha256
        authorization_audit_file_sha256 = Get-M43A4Sha256 $authorizationAuditPath
        plan_file_sha256 = $planFileSha256
        manifest_file_sha256 = $manifestFileSha256
        schedule_file_sha256 = $scheduleFileSha256
        result_objects_addressed_when_claimed = $false
        other_role_result_objects_addressed = $false
        current_profile_mutated = $false
        runtime_policy_activated = $false
        claimed_at_utc = (Get-Date).ToUniversalTime().ToString('o')
    }
    Write-M43A4Utf8CreateNew `
        -Path $RoleConsumptionMarkerPath `
        -Text (($claim | ConvertTo-Json -Depth 12) + "`n")
}
$claimFileSha256 = Get-M43A4Sha256 $RoleConsumptionMarkerPath

$copyJobs = @()
foreach ($spec in $roleSpecs) {
    $shardDir = Join-Path $shardsDir ('shard_{0:D3}' -f [int]$spec.shard)
    New-Item -ItemType Directory -Path $shardDir | Out-Null
    $roleResultPrefix = "$prefix/results/$($spec.output_prefix)"
    $copyJobs += [pscustomobject][ordered]@{
        label = "download Attempt04 $Role shard $($spec.shard)"
        destination = $shardDir
        sources = @(
            "$roleResultPrefix/teacher.jsonl",
            "$roleResultPrefix/DONE.json",
            "$roleResultPrefix/checkpoint.json",
            "$roleResultPrefix/heartbeat.json",
            "$roleResultPrefix/generator_summary.json",
            "$roleResultPrefix/run.log"
        )
    }
}
[void](Invoke-M43A4ParallelExactGcsCopies `
    -CopyJobs $copyJobs `
    -ProjectId $ProjectId `
    -MaxParallel $MaxParallel `
    -TimeoutSecondsPerJob 900)

$auditPaths = @()
$teacherPaths = @()
foreach ($spec in $roleSpecs) {
    $shardDir = Join-Path $shardsDir ('shard_{0:D3}' -f [int]$spec.shard)
    $auditPath = Join-Path $auditsDir ('shard_{0:D3}.json' -f [int]$spec.shard)
    Invoke-M43A4TrainingCli `
        -Arguments @(
            'validate-teacher-shard',
            '--plan', $PlanPath,
            '--repo-root', $RepoRoot,
            '--role', $Role,
            '--manifest', (Join-Path $closureDir 'manifest.json'),
            '--schedule', (Join-Path $closureDir 'shards_manifest.jsonl'),
            '--role-manifest', (Join-Path $closureDir 'teacher_role_manifest.json'),
            '--shard', ([string][int]$spec.shard),
            '--teacher', (Join-Path $shardDir 'teacher.jsonl'),
            '--done', (Join-Path $shardDir 'DONE.json'),
            '--checkpoint', (Join-Path $shardDir 'checkpoint.json'),
            '--heartbeat', (Join-Path $shardDir 'heartbeat.json'),
            '--generator-summary', (Join-Path $shardDir 'generator_summary.json'),
            '--run-log', (Join-Path $shardDir 'run.log'),
            '--output', $auditPath
        ) `
        -Label "validate Attempt04 $Role shard $($spec.shard)" `
        -TimeoutSeconds 600
    $audit = Get-Content -LiteralPath $auditPath -Raw | ConvertFrom-Json
    if ($audit.schema -ne 'hu_m43_attempt04_teacher_shard_audit_v1' -or
        $audit.status -ne 'pass' -or
        [string]$audit.logical_role -ne $Role -or
        [int]$audit.shard -ne [int]$spec.shard) {
        throw "Attempt04 shard audit did not pass: $($spec.shard)"
    }
    $auditPaths += $auditPath
    $teacherPaths += Join-Path $shardDir 'teacher.jsonl'
}

$mergedTeacherPath = Join-Path $mergedDir 'teacher.jsonl'
$receiptPath = Join-Path $mergedDir 'receive_receipt.json'
$mergeArguments = @(
    'merge-teacher-role',
    '--plan', $PlanPath,
    '--repo-root', $RepoRoot,
    '--role', $Role,
    '--manifest', (Join-Path $closureDir 'manifest.json'),
    '--schedule', (Join-Path $closureDir 'shards_manifest.jsonl'),
    '--role-manifest', (Join-Path $closureDir 'teacher_role_manifest.json'),
    '--authorization', $RoleOpenAuthorizationPath,
    '--consumption-marker', $RoleConsumptionMarkerPath,
    '--claim-file-sha256', $claimFileSha256,
    '--output', $mergedTeacherPath,
    '--receipt-output', $receiptPath,
    '--shard-audits'
) + $auditPaths + @('--teachers') + $teacherPaths
Invoke-M43A4TrainingCli `
    -Arguments $mergeArguments `
    -Label "merge Attempt04 $Role teacher role" `
    -TimeoutSeconds 900
if (-not (Test-Path -LiteralPath $mergedTeacherPath -PathType Leaf) -or
    -not (Test-Path -LiteralPath $receiptPath -PathType Leaf)) {
    throw 'Attempt04 role merge did not produce its teacher and receipt'
}
$receipt = Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json
if ($receipt.schema -ne 'hu_m43_attempt04_teacher_role_receive_receipt_v1' -or
    $receipt.status -ne 'verified_role_after_frozen_claim' -or
    [string]$receipt.logical_role -ne $Role -or
    [string]$receipt.role_identity_sha256 -ne $roleIdentitySha256 -or
    [string]$receipt.consumption_marker_file_sha256 -ne $claimFileSha256) {
    throw 'Attempt04 role receipt identity did not pass'
}
Assert-M43A4Sha256 ([string]$receipt.receipt_sha256) 'role receipt self hash'

Sync-M43A4TreeBestEffort -Path $stagingRoot
Publish-M43A4DirectoryAtomic -Source $stagingRoot -Destination $OutputDir
Sync-M43A4TreeBestEffort -Path $OutputDir
$localFinalization = Invoke-M43A4PostPublishFinalization `
    -OutputDirectory $OutputDir `
    -ClaimFileSha256 $claimFileSha256 `
    -RecoveredExistingOutput $false
[pscustomobject][ordered]@{
    schema = 'hu_m43_attempt04_teacher_role_receive_result_v1'
    status = 'published_verified_role'
    run_name = $RunName
    logical_role = $Role
    role_identity_sha256 = $roleIdentitySha256
    shards = $roleSpecs.Count
    roots = $roleSpecs.Count * 10
    output_dir = $OutputDir
    merged_teacher = Join-Path $OutputDir 'merged/teacher.jsonl'
    receipt = [string]$localFinalization.published_receipt
    receipt_file_sha256 = [string]$localFinalization.receipt_file_sha256
    published_audit = [string]$localFinalization.published_audit
    published_audit_file_sha256 = [string]$localFinalization.published_audit_file_sha256
    finalization = [string]$localFinalization.finalization
    finalization_file_sha256 = [string]$localFinalization.finalization_file_sha256
    consumption_marker = $RoleConsumptionMarkerPath
    consumption_marker_file_sha256 = $claimFileSha256
    parallel_download_limit = $MaxParallel
    resumed_existing_claim = [bool]$ResumeClaim
    other_role_result_objects_addressed = $false
    current_profile_mutated = $false
    runtime_policy_activated = $false
} | ConvertTo-Json -Depth 10
