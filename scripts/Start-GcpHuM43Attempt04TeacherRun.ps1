param(
    [string]$ProjectId = 'ofc-solver-485418',
    [string]$Bucket = 'pokerhu-ofc-solver-485418-training',
    [string]$RunName = ('regular-hu-m43-attempt04-c2e128-fresh700-' + (Get-Date -Format 'yyyyMMdd-HHmmss')),
    [string]$PlanPath = 'configs/hu_joint_policy_m43_attempt04.json',
    [Parameter(Mandatory = $true)][string]$ModelFreezePath,
    [Parameter(Mandatory = $true)][string]$TrainingFreezePath,
    [string]$TrainingModule = 'ofc_regular.hu_m43_attempt04_training',
    [string[]]$StartShards = @('0'),
    [string]$Zone = 'asia-northeast1-b',
    [string]$MachineType = 'c4-standard-4',
    [string]$BootDiskType = 'hyperdisk-balanced',
    [ValidateRange(20, 100)][int]$BootDiskGb = 50,
    [ValidateRange(1, 32)][int]$NativeBatchThreads = 4,
    [ValidateRange(10, 600)][int]$SyncIntervalSeconds = 60,
    [switch]$CreateInstances,
    [switch]$ResumeExisting,
    [switch]$SkipExistingInstances,
    [switch]$PackageOnly,
    [switch]$NoSelfDelete
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'HuM43Attempt04Spot.Common.ps1')

$PinnedTemplateRunName = 'regular-hu-m43-attempt03-c2e64-pilot700-r2-20260714-0228'
$PinnedTemplateManifestSha256 = '1e9cc09b3968322bb5b2eeb137a9ddc7aa067e9b8f3f8a2efd424daea2441aea'
$PinnedTemplateStartupSha256 = 'f42977efa91a1b3c543ed2cdc864b9f1712bc96047633b2217c46b80387d2316'
$PinnedTemplateModelManifestSha256 = 'e9b9d9748bb2b2f31a4baa0d928b60c05f254ab5d7915a7461fb8402e1e7ddb8'
$PinnedTemplateNativeManifestSha256 = 'ec295d070ac7bb21a82aeb2f432b8f6f191e61b63027bc446e8bd5f39f51569f'
$PinnedTemplateScheduleSha256 = '57e90d89d439ccfe67df78bd6787b075d32ba8b827ffef7dae88cc323664d923'
$PinnedTemplatePackageTreeSha256 = 'e16764d716b2c749900499667fd13b29bf9cd3472e85d62add6eb9703cf798e4'

$RoleOrder = @(
    'precal_holdout',
    'calibration_safety_fit',
    'calibration_threshold_lock',
    'locked_holdout'
)
$ExpectedRoots = [ordered]@{
    'precal_holdout' = 300
    'calibration_safety_fit' = 100
    'calibration_threshold_lock' = 100
    'locked_holdout' = 200
}
$ExpectedSplits = [ordered]@{
    'precal_holdout' = 'train'
    'calibration_safety_fit' = 'train'
    'calibration_threshold_lock' = 'train'
    'locked_holdout' = 'train'
}
$ExpectedSeedStarts = [ordered]@{
    'precal_holdout' = 13106071901L
    'calibration_safety_fit' = 13506071901L
    'calibration_threshold_lock' = 13606071901L
    'locked_holdout' = 13806071901L
}
$ExpectedCandidateStarts = [ordered]@{
    'precal_holdout' = 20106071901L
    'calibration_safety_fit' = 20506071901L
    'calibration_threshold_lock' = 20606071901L
    'locked_holdout' = 20806071901L
}
$ExpectedEvaluationStarts = [ordered]@{
    'precal_holdout' = 21106071901L
    'calibration_safety_fit' = 21506071901L
    'calibration_threshold_lock' = 21606071901L
    'locked_holdout' = 21806071901L
}
$ExpectedChildStarts = [ordered]@{
    'precal_holdout' = 22106071901L
    'calibration_safety_fit' = 22506071901L
    'calibration_threshold_lock' = 22606071901L
    'locked_holdout' = 22806071901L
}
$SeedStride = 1000003L
$RootsPerShard = 10
$TotalRoots = 700
$TotalShards = 70

if ($RunName -notmatch '^[a-z0-9][a-z0-9-]{2,126}[a-z0-9]$') {
    throw 'RunName is not a safe GCP run identity'
}
if ($TrainingModule -notmatch '^[A-Za-z_][A-Za-z0-9_.]*$') {
    throw 'TrainingModule is not a safe Python module name'
}
if ($PackageOnly -and $CreateInstances) {
    throw 'PackageOnly and CreateInstances are mutually exclusive'
}

function Expand-M43A4ShardSelection {
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

function ConvertTo-M43A4VmPrefix {
    param([string]$Value)
    $result = (($Value.ToLowerInvariant() -replace '[^a-z0-9-]', '-') -replace '-+', '-').Trim('-')
    if (-not $result) { throw 'RunName cannot produce a VM prefix' }
    if ($result.Length -gt 54) { $result = $result.Substring(0, 54).TrimEnd('-') }
    return $result
}

function Get-M43A4PackageTreeSha256 {
    param([Parameter(Mandatory = $true)][string]$Root)
    $fullRoot = [IO.Path]::GetFullPath($Root).TrimEnd(
        [IO.Path]::DirectorySeparatorChar,
        [IO.Path]::AltDirectorySeparatorChar
    )
    if (-not (Test-Path -LiteralPath $fullRoot -PathType Container)) {
        throw "Pinned template package tree is missing: $fullRoot"
    }
    $relativePaths = [Collections.Generic.List[string]]::new()
    foreach ($file in Get-ChildItem -LiteralPath $fullRoot -Recurse -File) {
        $relative = $file.FullName.Substring($fullRoot.Length + 1).Replace('\', '/')
        if ($relative -match "[`t`r`n]") {
            throw "Pinned template package path is not canonical: $relative"
        }
        $relativePaths.Add($relative)
    }
    $relativePaths.Sort([StringComparer]::Ordinal)
    $canonical = [Text.StringBuilder]::new()
    foreach ($relative in $relativePaths) {
        $nativeRelative = $relative.Replace(
            '/', [IO.Path]::DirectorySeparatorChar
        )
        $path = Join-Path $fullRoot $nativeRelative
        $length = (Get-Item -LiteralPath $path).Length
        [void]$canonical.Append($relative)
        [void]$canonical.Append("`t")
        [void]$canonical.Append([string]$length)
        [void]$canonical.Append("`t")
        [void]$canonical.Append((Get-M43A4Sha256 $path))
        [void]$canonical.Append("`n")
    }
    $bytes = [Text.UTF8Encoding]::new($false).GetBytes($canonical.ToString())
    $hasher = [Security.Cryptography.SHA256]::Create()
    try { $hash = $hasher.ComputeHash($bytes) }
    finally { $hasher.Dispose() }
    return ([BitConverter]::ToString($hash) -replace '-', '').ToLowerInvariant()
}

function Assert-M43A4PinnedTeacherTemplate {
    param([Parameter(Mandatory = $true)][string]$TemplateDirectory)
    $templateManifestPath = Join-Path $TemplateDirectory 'manifest.json'
    $templatePackagePath = Join-Path $TemplateDirectory 'package_src'
    $templateStartupPath = Join-Path $TemplateDirectory 'startup_hu_m43_attempt03_teacher.sh'
    $templateModelPath = Join-Path $templatePackagePath 'source_model_manifest.json'
    $templateNativePath = Join-Path $templatePackagePath 'source_native_manifest.json'
    $templateSchedulePath = Join-Path $templatePackagePath 'shards_manifest.jsonl'
    foreach ($path in @(
        $templateManifestPath,
        $templateStartupPath,
        $templateModelPath,
        $templateNativePath,
        $templateSchedulePath
    )) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Pinned Attempt03 teacher template input is missing: $path"
        }
    }
    if ((Get-M43A4Sha256 $templateManifestPath) -ne $PinnedTemplateManifestSha256 -or
        (Get-M43A4Sha256 $templateStartupPath) -ne $PinnedTemplateStartupSha256 -or
        (Get-M43A4Sha256 $templateModelPath) -ne $PinnedTemplateModelManifestSha256 -or
        (Get-M43A4Sha256 $templateNativePath) -ne $PinnedTemplateNativeManifestSha256 -or
        (Get-M43A4Sha256 $templateSchedulePath) -ne $PinnedTemplateScheduleSha256 -or
        (Get-M43A4PackageTreeSha256 $templatePackagePath) -ne $PinnedTemplatePackageTreeSha256) {
        throw 'Pinned Attempt03 teacher template byte closure changed'
    }
    $templateManifest = Get-Content -LiteralPath $templateManifestPath -Raw | ConvertFrom-Json
    if ($templateManifest.schema -ne 'hu_m43_attempt03_teacher_spot_manifest_v1' -or
        [string]$templateManifest.run_name -ne $PinnedTemplateRunName -or
        [string]$templateManifest.schedule_sha256 -ne $PinnedTemplateScheduleSha256 -or
        [string]$templateManifest.startup_sha256 -ne $PinnedTemplateStartupSha256 -or
        [string]$templateManifest.model_manifest_sha256 -ne $PinnedTemplateModelManifestSha256 -or
        [string]$templateManifest.native_manifest_sha256 -ne $PinnedTemplateNativeManifestSha256 -or
        [int]$templateManifest.candidate_samples -ne 2 -or
        [int]$templateManifest.evaluation_samples -ne 64) {
        throw 'Pinned Attempt03 teacher template semantic closure changed'
    }
}

function Assert-M43A4FreezeBoundary {
    param([string]$Path, [string]$Label)
    $payload = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    if ([string]$payload.schema -notmatch '^hu_m43_attempt04_' -or
        [string]$payload.status -notmatch '^frozen' -or
        ($null -ne $payload.PSObject.Properties['current_profile_mutated'] -and
            [bool]$payload.current_profile_mutated) -or
        ($null -ne $payload.PSObject.Properties['runtime_policy_activated'] -and
            [bool]$payload.runtime_policy_activated)) {
        throw "$Label is not a frozen, non-runtime Attempt04 artifact"
    }
}

function Get-M43A4StrictDone {
    param($Done, $Spec, $Manifest, [string]$ManifestSha256)
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
    )) { Assert-M43A4Sha256 $value 'teacher DONE artifact hash' }
    return $Done
}

function Get-M43A4RemoteJsonExact {
    param([string]$Uri, [string]$ExpectedSha256)
    $temporary = Join-Path ([IO.Path]::GetTempPath()) ('m43a4-json-' + [guid]::NewGuid().ToString('N') + '.json')
    try {
        Copy-M43A4RemoteFileExact -Uri $Uri -Destination $temporary -ProjectId $ProjectId -ExpectedSha256 $ExpectedSha256
        return (Get-Content -LiteralPath $temporary -Raw | ConvertFrom-Json)
    }
    finally { Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue }
}

function Get-M43A4RoleManifestClosure {
    param(
        [Parameter(Mandatory = $true)]$Contract,
        [Parameter(Mandatory = $true)][string]$RunDirectory
    )
    $properties = @($Contract.role_manifests.PSObject.Properties)
    if ((@($properties.Name | Sort-Object) -join '|') -ne
        (@($RoleOrder | Sort-Object) -join '|')) {
        throw 'Attempt04 contract role manifest set changed'
    }
    $closure = [ordered]@{}
    foreach ($role in $RoleOrder) {
        $binding = $Contract.role_manifests.PSObject.Properties[$role].Value
        $relative = [string]$binding.path
        $expectedRelative = 'teacher_role_manifests/' + $role + '.json'
        if ($relative -ne $expectedRelative) {
            throw "Attempt04 role manifest path changed: $role"
        }
        $localPath = [IO.Path]::GetFullPath((Join-Path $RunDirectory (
            $relative -replace '/', [IO.Path]::DirectorySeparatorChar
        )))
        Assert-M43A4UnderRoot `
            -Path $localPath -Root $RunDirectory -Label "Attempt04 $role manifest"
        if (-not (Test-Path -LiteralPath $localPath -PathType Leaf) -or
            (Get-M43A4Sha256 $localPath) -ne [string]$binding.file_sha256) {
            throw "Attempt04 role manifest file/hash changed: $role"
        }
        Assert-M43A4Sha256 ([string]$binding.role_identity_sha256) 'role identity hash'
        $payload = Get-Content -LiteralPath $localPath -Raw | ConvertFrom-Json
        if ($payload.schema -ne 'hu_m43_attempt04_ordered_teacher_roles_v1' -or
            $payload.status -ne 'frozen_pre_generation' -or
            [string]$payload.logical_role -ne $role -or
            [string]$payload.role_identity_sha256 -ne [string]$binding.role_identity_sha256) {
            throw "Attempt04 role manifest identity changed: $role"
        }
        $closure[$role] = [pscustomobject][ordered]@{
            path = $localPath
            relative_path = $relative
            file_sha256 = [string]$binding.file_sha256
            role_identity_sha256 = [string]$binding.role_identity_sha256
        }
    }
    return $closure
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$PlanPath = Resolve-M43A4Path -Path $PlanPath -Root $repoRoot -Label 'Attempt04 plan' -RequireFile
$ModelFreezePath = Resolve-M43A4Path -Path $ModelFreezePath -Root $repoRoot -Label 'Attempt04 model freeze' -RequireFile
$TrainingFreezePath = Resolve-M43A4Path -Path $TrainingFreezePath -Root $repoRoot -Label 'Attempt04 training freeze' -RequireFile
$TemplateRunDir = Resolve-M43A4Path `
    -Path (Join-Path 'outputs/gcp_runs' $PinnedTemplateRunName) `
    -Root $repoRoot `
    -Label 'pinned teacher template'
Assert-M43A4PinnedTeacherTemplate -TemplateDirectory $TemplateRunDir
Assert-M43A4FreezeBoundary $ModelFreezePath 'ModelFreezePath'
Assert-M43A4FreezeBoundary $TrainingFreezePath 'TrainingFreezePath'
$plan = Get-Content -LiteralPath $PlanPath -Raw | ConvertFrom-Json
if ($plan.schema -ne 'hu_m43_attempt04_plan_v1' -or $plan.status -ne 'frozen_pre_generation') {
    throw 'Attempt04 plan must be frozen before generation'
}
$splitProperties = @($plan.fresh_splits.PSObject.Properties)
if ((@($splitProperties.Name | Sort-Object) -join '|') -ne (@($RoleOrder | Sort-Object) -join '|')) {
    throw 'Attempt04 fresh split roles changed'
}
if ([int]$plan.teacher_search.candidate_samples -ne 2 -or
    [int]$plan.teacher_search.evaluation_samples -ne 128 -or
    -not [bool]$plan.teacher_search.common_random_futures -or
    -not [bool]$plan.teacher_search.candidate_evaluation_rng_disjoint) {
    throw 'Attempt04 c2/e128 search contract changed'
}
if ($NativeBatchThreads -ne [int]$plan.teacher_search.native_batch_threads) {
    throw 'NativeBatchThreads does not match the frozen plan'
}
foreach ($role in $RoleOrder) {
    $spec = $plan.fresh_splits.PSObject.Properties[$role].Value
    if ([int]$spec.roots -ne [int]$ExpectedRoots[$role] -or
        [string]$spec.record_split -ne [string]$ExpectedSplits[$role] -or
        [long]$spec.seed_start -ne [long]$ExpectedSeedStarts[$role] -or
        [long]$spec.seed_stride -ne $SeedStride -or
        [long]$spec.candidate_seed_start -ne [long]$ExpectedCandidateStarts[$role] -or
        [long]$spec.evaluation_seed_start -ne [long]$ExpectedEvaluationStarts[$role] -or
        [long]$spec.child_policy_seed_start -ne [long]$ExpectedChildStarts[$role]) {
        throw "Attempt04 frozen role changed: $role"
    }
}

$runRoot = Join-Path $repoRoot 'outputs/gcp_runs'
$runDir = Join-Path $runRoot $RunName
$manifestPath = Join-Path $runDir 'manifest.json'
$schedulePath = Join-Path $runDir 'shards_manifest.jsonl'
$contractPath = Join-Path $runDir 'teacher_contract.json'
$preflightPath = Join-Path $runDir 'teacher_preflight_receipt.json'
$sourcePath = Join-Path $runDir 'ofc_regular_hu_m43_attempt04_teacher_source.zip'
$startupPath = Join-Path $runDir 'startup_hu_m43_attempt04_teacher.sh'
$sourceModelPath = Join-Path $runDir 'source_model_manifest.json'
$sourceNativePath = Join-Path $runDir 'source_native_manifest.json'
$prefix = "gs://$Bucket/runs/$RunName"
$selection = @(Expand-M43A4ShardSelection -Values $StartShards -Count $TotalShards)

if ($ResumeExisting) {
    foreach ($path in @(
        $manifestPath, $schedulePath, $contractPath, $preflightPath, $sourcePath,
        $startupPath, $sourceModelPath, $sourceNativePath
    )) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Attempt04 resume artifact is missing: $path"
        }
    }
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    $contractPayload = Get-Content -LiteralPath $contractPath -Raw | ConvertFrom-Json
    if ($manifest.schema -ne 'hu_m43_attempt04_teacher_spot_manifest_v1' -or
        [string]$manifest.run_name -ne $RunName -or
        [string]$manifest.project_id -ne $ProjectId -or
        [string]$manifest.bucket -ne $Bucket -or
        [string]$manifest.model_freeze_file_sha256 -ne (Get-M43A4Sha256 $ModelFreezePath) -or
        [string]$manifest.training_freeze_file_sha256 -ne (Get-M43A4Sha256 $TrainingFreezePath)) {
        throw 'Attempt04 resume manifest identity changed'
    }
    if ($contractPayload.schema -ne 'hu_m43_attempt04_data_contract_v1' -or
        $contractPayload.status -ne 'frozen_pre_generation' -or
        [string]$contractPayload.plan_file_sha256 -ne (Get-M43A4Sha256 $PlanPath) -or
        [string]$contractPayload.model_freeze_file_sha256 -ne (Get-M43A4Sha256 $ModelFreezePath) -or
        [string]$contractPayload.training_freeze_file_sha256 -ne (Get-M43A4Sha256 $TrainingFreezePath)) {
        throw 'Attempt04 resume teacher contract binding changed'
    }
    $roleManifestClosure = Get-M43A4RoleManifestClosure `
        -Contract $contractPayload -RunDirectory $runDir
    foreach ($role in $RoleOrder) {
        $roleBinding = $roleManifestClosure[$role]
        $manifestBinding = $manifest.role_manifests.PSObject.Properties[$role].Value
        if ([string]$manifestBinding.file_sha256 -ne [string]$roleBinding.file_sha256 -or
            [string]$manifestBinding.role_identity_sha256 -ne [string]$roleBinding.role_identity_sha256 -or
            [string]$manifestBinding.gcs_path -ne ('source/teacher_role_manifests/' + $role + '.json')) {
            throw "Attempt04 resume role manifest binding changed: $role"
        }
    }
    foreach ($binding in @(
        @($schedulePath, $manifest.schedule_sha256),
        @($contractPath, $manifest.teacher_contract_file_sha256),
        @($preflightPath, $manifest.preflight_file_sha256),
        @($sourcePath, $manifest.source_sha256),
        @($startupPath, $manifest.startup_sha256),
        @($sourceModelPath, $manifest.model_manifest_sha256),
        @($sourceNativePath, $manifest.native_manifest_sha256)
    )) {
        if ((Get-M43A4Sha256 ([string]$binding[0])) -ne [string]$binding[1]) {
            throw "Attempt04 resume artifact hash changed: $($binding[0])"
        }
    }
    $remoteManifest = Join-Path ([IO.Path]::GetTempPath()) ('m43a4-manifest-' + [guid]::NewGuid().ToString('N') + '.json')
    try {
        Copy-M43A4RemoteFileExact `
            -Uri "$prefix/manifest.json" `
            -Destination $remoteManifest `
            -ProjectId $ProjectId `
            -ExpectedSha256 (Get-M43A4Sha256 $manifestPath)
    }
    finally { Remove-Item -LiteralPath $remoteManifest -Force -ErrorAction SilentlyContinue }
    foreach ($role in $RoleOrder) {
        $remoteRoleManifest = Join-Path (
            [IO.Path]::GetTempPath()
        ) ('m43a4-role-' + $role + '-' + [guid]::NewGuid().ToString('N') + '.json')
        try {
            Copy-M43A4RemoteFileExact `
                -Uri "$prefix/source/teacher_role_manifests/$role.json" `
                -Destination $remoteRoleManifest `
                -ProjectId $ProjectId `
                -ExpectedSha256 ([string]$roleManifestClosure[$role].file_sha256)
        }
        finally {
            Remove-Item -LiteralPath $remoteRoleManifest -Force -ErrorAction SilentlyContinue
        }
    }
}
else {
    if (Test-Path -LiteralPath $runDir) { throw 'Attempt04 RunName is immutable and already exists' }
    New-Item -ItemType Directory -Path $runDir | Out-Null
    $pythonPath = [string](Get-Command python -ErrorAction Stop | Select-Object -First 1).Source
    $contractResult = Invoke-M43A4ProcessBounded `
        -FilePath $pythonPath `
        -Arguments @(
            '-B', '-m', $TrainingModule, 'build-teacher-contract',
            '--plan', $PlanPath, '--repo-root', $repoRoot,
            '--model-freeze', $ModelFreezePath,
            '--training-freeze', $TrainingFreezePath,
            '--output', $contractPath, '--preflight-output', $preflightPath
        ) `
        -TimeoutSeconds 300 `
        -Label 'Attempt04 teacher contract preflight' `
        -Environment @{ PYTHONPATH = (Join-Path $repoRoot 'src'); PYTHONHASHSEED = '0' }
    if ($contractResult.timed_out -or $contractResult.exit_code -ne 0) {
        throw "Attempt04 teacher contract preflight failed: $($contractResult.stderr)$($contractResult.stdout)"
    }
    if (-not (Test-Path -LiteralPath $contractPath -PathType Leaf) -or
        -not (Test-Path -LiteralPath $preflightPath -PathType Leaf)) {
        throw 'Attempt04 teacher contract/preflight output is incomplete'
    }
    $contractPayload = Get-Content -LiteralPath $contractPath -Raw | ConvertFrom-Json
    $preflightPayload = Get-Content -LiteralPath $preflightPath -Raw | ConvertFrom-Json
    $planSha256 = Get-M43A4Sha256 $PlanPath
    $modelFreezeSha256 = Get-M43A4Sha256 $ModelFreezePath
    $trainingFreezeSha256 = Get-M43A4Sha256 $TrainingFreezePath
    if ($contractPayload.schema -ne 'hu_m43_attempt04_data_contract_v1' -or
        $contractPayload.status -ne 'frozen_pre_generation' -or
        $preflightPayload.schema -ne 'hu_m43_attempt04_teacher_preflight_receipt_v1' -or
        $preflightPayload.status -ne 'pass_frozen_before_fresh_generation') {
        throw 'Attempt04 teacher contract/preflight schema or status changed'
    }
    foreach ($payload in @($contractPayload, $preflightPayload)) {
        if ([string]$payload.plan_file_sha256 -ne $planSha256 -or
            [string]$payload.model_freeze_file_sha256 -ne $modelFreezeSha256 -or
            [string]$payload.training_freeze_file_sha256 -ne $trainingFreezeSha256) {
            throw 'Attempt04 teacher preflight lost its plan/model/training freeze binding'
        }
    }
    $roleManifestClosure = Get-M43A4RoleManifestClosure `
        -Contract $contractPayload -RunDirectory $runDir

    $specs = [Collections.Generic.List[object]]::new()
    $globalShard = 0
    foreach ($role in $RoleOrder) {
        $roleSpec = $plan.fresh_splits.PSObject.Properties[$role].Value
        $roleShards = [int]$roleSpec.roots / $RootsPerShard
        for ($roleShard = 0; $roleShard -lt $roleShards; $roleShard++) {
            $offset = $roleShard * $RootsPerShard
            $slug = ($role -replace '[^a-zA-Z0-9]+', '_').Trim('_').ToLowerInvariant()
            $seedStart = [long]$roleSpec.seed_start + [long]$offset * $SeedStride
            $specs.Add([ordered]@{
                schema = 'hu_m43_attempt04_teacher_shard_v1'
                shard = $globalShard
                logical_role = $role
                split = [string]$roleSpec.record_split
                role_shard = $roleShard
                roots = $RootsPerShard
                seed_start = $seedStart
                seed_stride = $SeedStride
                candidate_seed = [long]$roleSpec.candidate_seed_start + [long]$roleShard * $SeedStride
                evaluation_seed = [long]$roleSpec.evaluation_seed_start + [long]$roleShard * $SeedStride
                child_policy_seed = [long]$roleSpec.child_policy_seed_start + [long]$roleShard * $SeedStride
                candidate_samples = 2
                evaluation_samples = 128
                output_prefix = ('{0}_shard_{1:D3}_roots10_seed{2}' -f $slug, $roleShard, $seedStart)
                profile_quota_per_shard = [ordered]@{
                    stage19_p0 = 2
                    stage9f_p2 = 2
                    stage7_m5_r10 = 2
                    stage3_baseline = 2
                    random_exact_final = 2
                }
            })
            $globalShard++
        }
    }
    if ($specs.Count -ne $TotalShards) { throw 'Attempt04 schedule is not exact 70 shards' }
    $scheduleText = (@($specs | ForEach-Object { $_ | ConvertTo-Json -Compress -Depth 8 }) -join "`n") + "`n"
    Write-M43A4Utf8CreateNew $schedulePath $scheduleText

    $templatePackage = Join-Path $TemplateRunDir 'package_src'
    $templateStartup = Join-Path $TemplateRunDir 'startup_hu_m43_attempt03_teacher.sh'
    if (-not (Test-Path -LiteralPath $templatePackage -PathType Container) -or
        -not (Test-Path -LiteralPath $templateStartup -PathType Leaf)) {
        throw 'Pinned Attempt03 teacher closure template is missing'
    }
    $packageRoot = Join-Path $runDir 'package_src'
    Copy-Item -LiteralPath $templatePackage -Destination $packageRoot -Recurse
    $packagedSchedulePath = Join-Path $packageRoot 'shards_manifest.jsonl'
    Copy-Item -LiteralPath $schedulePath -Destination $packagedSchedulePath -Force
    if ((Get-M43A4Sha256 $packagedSchedulePath) -ne (Get-M43A4Sha256 $schedulePath)) {
        throw 'Attempt04 packaged schedule does not match the frozen schedule'
    }
    Copy-Item -LiteralPath $templateStartup -Destination $startupPath
    $startupText = [IO.File]::ReadAllText($startupPath)
    $startupReplacements = [ordered]@{
        'gcloud storage cp "$SYNC/"* "$PREFIX/resume/${OUTPUT_PREFIX}/" >/dev/null; status running; }' = @'
gcloud storage cp "$SYNC/teacher.jsonl.partial" "$PREFIX/resume/${OUTPUT_PREFIX}/teacher.jsonl.partial" >/dev/null; gcloud storage cp "$SYNC/checkpoint.json" "$PREFIX/resume/${OUTPUT_PREFIX}/checkpoint.json" >/dev/null; if [[ -s "$SYNC/heartbeat.json" ]]; then gcloud storage cp "$SYNC/heartbeat.json" "$PREFIX/resume/${OUTPUT_PREFIX}/heartbeat.json" >/dev/null; fi; status running; }
'@
        'if gcloud storage ls "$DONE_URI" >/dev/null 2>&1; then' = 'if gcloud storage objects describe "$DONE_URI" >/dev/null 2>&1; then'
        'if gcloud storage ls "$RESUME_URI/checkpoint.json" >/dev/null 2>&1; then' = 'if gcloud storage objects describe "$RESUME_URI/checkpoint.json" >/dev/null 2>&1; then'
    }
    foreach ($replacement in $startupReplacements.GetEnumerator()) {
        if (-not $startupText.Contains([string]$replacement.Key)) {
            throw "Attempt04 startup template replacement anchor changed: $($replacement.Key)"
        }
        $startupText = $startupText.Replace(
            [string]$replacement.Key,
            ([string]$replacement.Value).TrimEnd("`r", "`n")
        )
    }
    if ($startupText.Contains('gcloud storage ls ') -or
        $startupText.Contains('"$SYNC/"*')) {
        throw 'Attempt04 startup retained wildcard or list-based result access'
    }
    [IO.File]::WriteAllText(
        $startupPath,
        $startupText,
        [Text.UTF8Encoding]::new($false)
    )
    Copy-Item -LiteralPath (Join-Path $packageRoot 'source_model_manifest.json') -Destination $sourceModelPath
    Copy-Item -LiteralPath (Join-Path $packageRoot 'source_native_manifest.json') -Destination $sourceNativePath
    $zipBuilder = @'
import pathlib,sys,zipfile
root=pathlib.Path(sys.argv[1]).resolve(); destination=pathlib.Path(sys.argv[2]).resolve()
schedule=pathlib.Path(sys.argv[3]).resolve().read_bytes()
with zipfile.ZipFile(destination,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        name=path.relative_to(root).as_posix()
        if "\\" in name or name.startswith(("/","../")): raise ValueError(name)
        z.write(path,name)
with zipfile.ZipFile(destination,"r") as z:
    names=z.namelist()
    if names.count("shards_manifest.jsonl") != 1: raise ValueError("stale or duplicate schedule")
    if z.read("shards_manifest.jsonl") != schedule: raise ValueError("packaged schedule bytes changed")
'@
    $zipResult = Invoke-M43A4ProcessBounded `
        -FilePath $pythonPath -Arguments @(
            '-B', '-c', $zipBuilder, $packageRoot, $sourcePath, $schedulePath
        ) `
        -TimeoutSeconds 300 -Label 'Attempt04 teacher source archive'
    if ($zipResult.timed_out -or $zipResult.exit_code -ne 0) {
        throw "Attempt04 teacher archive failed: $($zipResult.stderr)$($zipResult.stdout)"
    }
    $roleManifestBindings = [ordered]@{}
    foreach ($role in $RoleOrder) {
        $roleBinding = $roleManifestClosure[$role]
        $roleManifestBindings[$role] = [ordered]@{
            file_sha256 = [string]$roleBinding.file_sha256
            role_identity_sha256 = [string]$roleBinding.role_identity_sha256
            gcs_path = 'source/teacher_role_manifests/' + $role + '.json'
        }
    }
    $manifest = [ordered]@{
        schema = 'hu_m43_attempt04_teacher_spot_manifest_v1'
        status = 'frozen'
        run_name = $RunName
        project_id = $ProjectId
        bucket = $Bucket
        plan_file_sha256 = Get-M43A4Sha256 $PlanPath
        model_freeze_file_sha256 = Get-M43A4Sha256 $ModelFreezePath
        training_freeze_file_sha256 = Get-M43A4Sha256 $TrainingFreezePath
        teacher_contract_file_sha256 = Get-M43A4Sha256 $contractPath
        preflight_file_sha256 = Get-M43A4Sha256 $preflightPath
        total_roots = $TotalRoots
        total_shards = $TotalShards
        roots_per_shard = $RootsPerShard
        role_roots = $ExpectedRoots
        role_manifests = $roleManifestBindings
        candidate_samples = 2
        evaluation_samples = 128
        schedule_sha256 = Get-M43A4Sha256 $schedulePath
        source_sha256 = Get-M43A4Sha256 $sourcePath
        startup_sha256 = Get-M43A4Sha256 $startupPath
        model_manifest_sha256 = Get-M43A4Sha256 $sourceModelPath
        native_manifest_sha256 = Get-M43A4Sha256 $sourceNativePath
        inherited_teacher_closure = [ordered]@{
            source_run = 'regular-hu-m43-attempt03-c2e64-pilot700-r2-20260714-0228'
            search_algorithm_unchanged = $true
            schedule_replaced_before_archive = $true
        }
        machine_type = $MachineType
        zone = $Zone
        spot = $true
        self_delete = (-not [bool]$NoSelfDelete)
        result_content_opened = $false
        current_profile_mutated = $false
        runtime_policy_activated = $false
        full_replacement = $false
        created_at = (Get-Date).ToUniversalTime().ToString('o')
    }
    Write-M43A4Utf8CreateNew $manifestPath (($manifest | ConvertTo-Json -Depth 20) + "`n")
    if (-not $PackageOnly) {
        if (Test-M43A4GcsObject -Uri "$prefix/manifest.json" -ProjectId $ProjectId) {
            throw 'Attempt04 remote teacher run already exists'
        }
        $uploadBindings = @(
            @($sourcePath, "$prefix/source/ofc_regular_hu_m43_attempt04_teacher_source.zip"),
            @($startupPath, "$prefix/source/startup_hu_m43_attempt04_teacher.sh"),
            @($schedulePath, "$prefix/source/shards_manifest.jsonl"),
            @($contractPath, "$prefix/source/teacher_contract.json"),
            @($preflightPath, "$prefix/source/teacher_preflight_receipt.json"),
            @($sourceModelPath, "$prefix/source/source_model_manifest.json"),
            @($sourceNativePath, "$prefix/source/source_native_manifest.json")
        )
        foreach ($role in $RoleOrder) {
            $roleBinding = $roleManifestClosure[$role]
            $uploadBindings += ,@(
                [string]$roleBinding.path,
                "$prefix/source/teacher_role_manifests/$role.json"
            )
        }
        $uploadBindings += ,@($manifestPath, "$prefix/manifest.json")
        foreach ($binding in $uploadBindings) {
            [void](Invoke-M43A4Gcloud `
                -Arguments @('storage', 'cp', $binding[0], $binding[1], '--project', $ProjectId, '--if-generation-match=0') `
                -TimeoutSeconds 300 `
                -Label "upload $($binding[1])")
        }
    }
}

if ($PackageOnly) {
    [pscustomobject][ordered]@{
        schema = 'hu_m43_attempt04_teacher_package_result_v1'
        status = 'packaged_no_cloud'
        run_name = $RunName
        manifest = $manifestPath
        manifest_sha256 = Get-M43A4Sha256 $manifestPath
        shards = $TotalShards
        roots = $TotalRoots
        current_profile_mutated = $false
    } | ConvertTo-Json -Depth 8
    exit 0
}
if (-not $CreateInstances) {
    [pscustomobject][ordered]@{
        schema = 'hu_m43_attempt04_teacher_start_result_v1'
        status = 'uploaded_no_instances'
        run_name = $RunName
        manifest_sha256 = Get-M43A4Sha256 $manifestPath
        current_profile_mutated = $false
    } | ConvertTo-Json -Depth 8
    exit 0
}

$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$specs = @([IO.File]::ReadLines($schedulePath) | ForEach-Object { $_ | ConvertFrom-Json })
$manifestSha = Get-M43A4Sha256 $manifestPath
if (@($selection | Where-Object { $_ -gt 0 }).Count -gt 0) {
    $canarySpec = $specs[0]
    $canaryUri = "$prefix/results/$($canarySpec.output_prefix)/DONE.json"
    if (-not (Test-M43A4GcsObject -Uri $canaryUri -ProjectId $ProjectId)) {
        throw 'Attempt04 shard0 metadata canary must complete before fanout'
    }
    $canary = Get-M43A4RemoteJsonExact -Uri $canaryUri
    [void](Get-M43A4StrictDone -Done $canary -Spec $canarySpec -Manifest $manifest -ManifestSha256 $manifestSha)
}

$vmPrefix = ConvertTo-M43A4VmPrefix $RunName
$started = @()
foreach ($shard in $selection) {
    $spec = $specs[$shard]
    $doneUri = "$prefix/results/$($spec.output_prefix)/DONE.json"
    if (Test-M43A4GcsObject -Uri $doneUri -ProjectId $ProjectId) {
        $done = Get-M43A4RemoteJsonExact -Uri $doneUri
        [void](Get-M43A4StrictDone -Done $done -Spec $spec -Manifest $manifest -ManifestSha256 $manifestSha)
        continue
    }
    $vmName = ('{0}-{1:D3}' -f $vmPrefix, $shard)
    $describe = Invoke-M43A4GcloudProcess `
        -Arguments @('compute', 'instances', 'describe', $vmName, '--project', $ProjectId, '--zone', $Zone, '--format=json') `
        -TimeoutSeconds 30 -Label "describe $vmName"
    if (-not $describe.timed_out -and $describe.exit_code -eq 0) {
        if (-not $SkipExistingInstances) { throw "Attempt04 teacher worker exists: $vmName" }
        continue
    }
    if ($describe.timed_out -or (([string]$describe.stdout + [string]$describe.stderr) -notmatch '(?i)not found|404')) {
        throw "Unable to prove worker absence: $vmName"
    }
    $selfDeleteValue = '1'
    if ($NoSelfDelete) { $selfDeleteValue = '0' }
    $metadata = @(
        "RUN_NAME=$RunName", "BUCKET=$Bucket", "SHARD_INDEX=$shard",
        "SOURCE_URI=$prefix/source/ofc_regular_hu_m43_attempt04_teacher_source.zip",
        "SOURCE_SHA256=$($manifest.source_sha256)", "STARTUP_SHA256=$($manifest.startup_sha256)",
        "MANIFEST_SHA256=$manifestSha", "SHARDS_SHA256=$($manifest.schedule_sha256)",
        "MODELS_SHA256=$($manifest.model_manifest_sha256)", "NATIVE_SHA256=$($manifest.native_manifest_sha256)",
        "SYNC_INTERVAL_SECONDS=$SyncIntervalSeconds", "NATIVE_BATCH_THREADS=$NativeBatchThreads",
        "SELF_DELETE=$selfDeleteValue"
    ) -join ','
    $arguments = @(
        'compute', 'instances', 'create', $vmName, '--project', $ProjectId, '--zone', $Zone,
        '--machine-type', $MachineType, '--image-family', 'ubuntu-2404-lts-amd64',
        '--image-project', 'ubuntu-os-cloud', '--boot-disk-size', ("${BootDiskGb}GB"),
        '--boot-disk-type', $BootDiskType, '--boot-disk-auto-delete', '--provisioning-model', 'SPOT',
        '--instance-termination-action', 'DELETE', '--maintenance-policy', 'TERMINATE',
        '--scopes', 'cloud-platform', '--labels', 'purpose=hu-m43-a04-teacher,milestone=m43-a04',
        '--metadata', $metadata, '--metadata-from-file', "startup-script=$startupPath",
        '--async', '--format=value(name)', '--quiet'
    )
    $create = Invoke-M43A4GcloudProcess -Arguments $arguments -TimeoutSeconds 45 -Label "create $vmName"
    if ($create.timed_out -or $create.exit_code -ne 0) {
        throw "Unable to submit Attempt04 teacher worker $vmName`: $($create.stderr)$($create.stdout)"
    }
    $started += [pscustomobject][ordered]@{
        shard = $shard
        logical_role = [string]$spec.logical_role
        name = $vmName
        zone = $Zone
        machine_type = $MachineType
    }
}
[pscustomobject][ordered]@{
    schema = 'hu_m43_attempt04_teacher_start_result_v1'
    status = 'submitted'
    run_name = $RunName
    selected_shards = $selection
    submitted = $started
    manifest_sha256 = $manifestSha
    spot = $true
    result_content_opened = $false
    current_profile_mutated = $false
    runtime_policy_activated = $false
} | ConvertTo-Json -Depth 10
