param(
    [Parameter(Mandatory = $true)][string]$ModelPath,
    [Parameter(Mandatory = $true)][string]$TrainingManifestPath,
    [Parameter(Mandatory = $true)][string]$RuntimeFreezePath,
    [Parameter(Mandatory = $true)][string]$RuntimeSourceArchivePath,
    [Parameter(Mandatory = $true)][string]$RuntimeSourceManifestPath,
    [string]$FrozenRuntimeArtifactRootPath = "",
    [string]$PlanPath = "configs/hu_joint_policy_m43_attempt10_population.json",
    [string]$ProjectId = "ofc-solver-485418",
    [string]$Bucket = "pokerhu-ofc-solver-485418-training",
    [string]$RunName = ("regular-hu-m43-attempt10-population-" + (Get-Date -Format "yyyyMMdd-HHmmss")),
    [string]$MachineType = "c4-standard-4",
    [string[]]$FallbackMachineTypes = @("n2-standard-4", "e2-standard-4"),
    [string[]]$Zones = @("asia-northeast1-b", "asia-northeast1-c", "us-central1-a"),
    [int]$BootDiskGb = 30,
    [int]$SyncIntervalSeconds = 60,
    [string[]]$StartShards = @(),
    [switch]$CreateInstances,
    [switch]$ResumeExisting,
    [switch]$SkipExistingInstances,
    [switch]$NoSelfDelete,
    [switch]$PackageOnly,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$ManifestSchema = "hu_m43_attempt10_population_spot_manifest_v1"
$PreflightSchema = "hu_m43_attempt10_population_launch_preflight_v1"
$DoneSchema = "hu_m43_attempt10_population_spot_done_v1"
$StatusSchema = "hu_m43_attempt10_population_spot_status_v1"
$ActionScoreMode = "attempt10_lambda_top12_distilled_safe_selector_v1"
$ModelSchema = "hu_m43_attempt10_t1_second_distilled_selector_v1"
$ArtifactSchema = "hu_m43_attempt10_t1_second_distilled_pickle_v1"
$FeatureSchema = "hu_m43_attempt10_lambda_top12_public_infoset_features_v1"
$HeadSchema = "hu_m43_attempt10_policy_delta_safe_tail_heads_v1"
$BaselineProfile = "stage19_p0"
$PinnedImage = "projects/debian-cloud/global/images/debian-12-bookworm-v20260609"
$PinnedImageId = "1449487925682397051"
$RuntimeRequirementsSha256 = "a0ce16d1ab481528ac53f2e94fd9037af7f94004a38b8cf8116537bbf4277c68"
$RuntimeFingerprintSha256 = "8c2cd111bc4e70096ff4f974f684ad146e94329871328e5b5db5d3426256c218"
$PinnedRuntimeTemplateRun = "regular-hu-m43-attempt03-c2e64-pilot700-r2-20260714-0228"
$PinnedModelManifestSha256 = "e9b9d9748bb2b2f31a4baa0d928b60c05f254ab5d7915a7461fb8402e1e7ddb8"
$PinnedNativeManifestSha256 = "ec295d070ac7bb21a82aeb2f432b8f6f191e61b63027bc446e8bd5f39f51569f"
$PackageReadySchema = "hu_m43_attempt10_population_package_ready_v1"

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}
function Assert-Sha256Text($Value, [string]$Label) {
    if ($Value -isnot [string] -or $Value -cnotmatch '^[0-9a-f]{64}$') {
        throw "$Label must be a lowercase SHA-256 digest"
    }
}
function Write-Utf8NoBom([string]$Path, [string]$Text) {
    [System.IO.File]::WriteAllText($Path, $Text, [System.Text.UTF8Encoding]::new($false))
}
function Write-NewUtf8NoBom([string]$Path, [string]$Text) {
    $stream = [System.IO.File]::Open($Path, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
    try {
        $encoded = [System.Text.UTF8Encoding]::new($false).GetBytes($Text)
        $stream.Write($encoded, 0, $encoded.Length)
        $stream.Flush($true)
    }
    finally { $stream.Dispose() }
}
function Invoke-Gcloud([string[]]$Arguments) {
    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { $output = & gcloud @Arguments 2>&1; $code = $LASTEXITCODE }
    finally { $ErrorActionPreference = $old }
    if ($code -ne 0) {
        throw "gcloud failed ($code): gcloud $($Arguments -join ' ')`n$(@($output) -join [Environment]::NewLine)"
    }
    return @($output)
}
function Test-GcsObject([string]$Uri) {
    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { $output = & gcloud storage objects describe $Uri --format=json 2>&1; $code = $LASTEXITCODE }
    finally { $ErrorActionPreference = $old }
    if ($code -eq 0) { return $true }
    if ((@($output) -join "`n") -match '(?i)not found|does not exist|No URLs matched|404') { return $false }
    throw "Could not inspect immutable GCS object: $Uri`n$(@($output) -join [Environment]::NewLine)"
}
function Publish-ImmutableObject([string]$LocalPath, [string]$Uri) {
    if (-not (Test-GcsObject $Uri)) {
        Invoke-Gcloud @("storage", "cp", $LocalPath, $Uri, "--project", $ProjectId, "--if-generation-match=0") | Out-Null
        return
    }
    $download = [System.IO.Path]::GetTempFileName()
    try {
        Invoke-Gcloud @("storage", "cp", $Uri, $download, "--project", $ProjectId) | Out-Null
        if ((Get-Sha256 $download) -ne (Get-Sha256 $LocalPath)) {
            throw "Immutable Attempt10 population object already exists with different bytes: $Uri"
        }
    }
    finally { Remove-Item -LiteralPath $download -Force -ErrorAction SilentlyContinue }
}
function Assert-LinuxX8664Elf([string]$Path) {
    $stream = [System.IO.File]::OpenRead($Path)
    try {
        $header = New-Object byte[] 20
        if ($stream.Read($header, 0, $header.Length) -ne $header.Length -or
            $header[0] -ne 0x7f -or $header[1] -ne 0x45 -or
            $header[2] -ne 0x4c -or $header[3] -ne 0x46 -or
            $header[4] -ne 2 -or $header[5] -ne 1 -or
            $header[16] -ne 0x03 -or $header[17] -ne 0x00 -or
            $header[18] -ne 0x3e -or $header[19] -ne 0x00) {
            throw "Population native dependency is not Linux x86_64 ELF: $Path"
        }
    }
    finally { $stream.Dispose() }
}
function New-ZipWithForwardSlashes([string]$SourceDir, [string]$DestinationPath) {
    Add-Type -AssemblyName System.IO.Compression
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    if (Test-Path -LiteralPath $DestinationPath) { Remove-Item -LiteralPath $DestinationPath -Force }
    $sourceRoot = (Resolve-Path -LiteralPath $SourceDir).Path.TrimEnd('\', '/')
    $prefixLength = $sourceRoot.Length + 1
    $zip = [System.IO.Compression.ZipFile]::Open($DestinationPath, [System.IO.Compression.ZipArchiveMode]::Create)
    try {
        Get-ChildItem -LiteralPath $sourceRoot -Recurse -File | Sort-Object FullName | ForEach-Object {
            $entry = $_.FullName.Substring($prefixLength) -replace '\\', '/'
            [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
                $zip, $_.FullName, $entry, [System.IO.Compression.CompressionLevel]::Optimal
            ) | Out-Null
        }
    }
    finally { $zip.Dispose() }
}
function Convert-ToVmPrefix([string]$Value) {
    $name = ($Value.ToLowerInvariant() -replace '[^a-z0-9-]', '-').Trim('-')
    if (-not $name) { throw "RunName does not produce a VM prefix" }
    if ($name -notmatch '^[a-z]') { $name = "m43a8p-$name" }
    if ($name.Length -gt 56) { $name = $name.Substring(0, 56).TrimEnd('-') }
    return $name
}

if ($RunName -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$') { throw "RunName contains unsafe characters" }
if ($BootDiskGb -lt 20) { throw "BootDiskGb must be at least 20" }
if ($SyncIntervalSeconds -lt 15) { throw "SyncIntervalSeconds must be at least 15" }
if (-not $Zones -or $Zones.Count -eq 0) { throw "At least one zone is required" }
if ($DryRun -and $CreateInstances) { throw "DryRun never creates cloud instances" }
if ($PackageOnly -and $CreateInstances) { throw "PackageOnly never creates cloud instances" }
if ($ResumeExisting -and $PackageOnly) { throw "ResumeExisting cannot repackage a frozen run" }
if ($CreateInstances -and -not $ResumeExisting) {
    throw "Fresh CreateInstances is forbidden: run -PackageOnly first, then use -ResumeExisting -CreateInstances with its immutable PACKAGE_READY marker"
}
if (-not $ResumeExisting -and -not $DryRun -and -not $PackageOnly) {
    throw "Fresh Attempt10 population mutation requires -PackageOnly; then use -ResumeExisting with its immutable PACKAGE_READY marker"
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
function Resolve-RepoPath([string]$Value) {
    $candidate = if ([System.IO.Path]::IsPathRooted($Value)) { $Value } else { Join-Path $repoRoot $Value }
    return (Resolve-Path -LiteralPath $candidate).Path
}

$runDir = Join-Path $repoRoot "outputs/gcp_runs/$RunName"
$packageDir = Join-Path $runDir "package"
$sourcePath = Join-Path $runDir "ofc_regular_hu_m43_attempt10_population_source.zip"
$manifestPath = Join-Path $runDir "population_run_manifest.json"
$shardsPath = Join-Path $runDir "population_shards.jsonl"
$startupPath = Join-Path $runDir "startup_hu_m43_attempt10_population.sh"
$packageReadyPath = Join-Path $runDir "PACKAGE_READY.json"
$gcsPrefix = "gs://$Bucket/runs/$RunName"
$sourceUri = "$gcsPrefix/source/ofc_regular_hu_m43_attempt10_population_source.zip"
$manifestUri = "$gcsPrefix/source/population_run_manifest.json"
$shardsUri = "$gcsPrefix/source/population_shards.jsonl"
$startupUri = "$gcsPrefix/source/startup_hu_m43_attempt10_population.sh"
$planUri = "$gcsPrefix/source/population_plan.json"
$modelUri = "$gcsPrefix/source/acceptance/m43_model.pkl"
$trainingManifestUri = "$gcsPrefix/source/acceptance/training_manifest.json"
$runtimeFreezeUri = "$gcsPrefix/source/acceptance/runtime_freeze.json"
$runtimeSourceArchiveUri = "$gcsPrefix/source/acceptance/runtime_source.zip"
$runtimeSourceManifestUri = "$gcsPrefix/source/acceptance/runtime_source_manifest.json"
$vmPrefix = Convert-ToVmPrefix $RunName

if ($ResumeExisting) {
    foreach ($path in @($manifestPath, $sourcePath, $shardsPath, $startupPath, $packageReadyPath)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Frozen Attempt10 population input is missing: $path" }
    }
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    $packageReady = Get-Content -LiteralPath $packageReadyPath -Raw | ConvertFrom-Json
    if ($manifest.schema -ne $ManifestSchema -or $manifest.run_name -ne $RunName -or
        $manifest.project_id -ne $ProjectId -or $manifest.bucket -ne $Bucket -or
        $manifest.no_runtime_activation -ne $true -or $manifest.current_profile_mutated -ne $false -or
        $manifest.runtime.model_schema -ne $ModelSchema -or
        $manifest.runtime.artifact_schema -ne $ArtifactSchema -or
        $manifest.runtime.feature_schema -ne $FeatureSchema -or
        $manifest.runtime.head_schema -ne $HeadSchema -or
        $manifest.runtime.action_score_mode -ne $ActionScoreMode -or
        $manifest.runtime.baseline_profile -ne $BaselineProfile -or
        $manifest.acceptance_artifacts.model.sha256 -ne $manifest.runtime.model_sha256 -or
        $manifest.acceptance_artifacts.training_manifest.sha256 -ne $manifest.runtime.training_manifest_sha256 -or
        $manifest.acceptance_artifacts.runtime_freeze.sha256 -ne $manifest.runtime.runtime_freeze_sha256 -or
        $manifest.acceptance_artifacts.runtime_source_archive.sha256 -ne $manifest.runtime.runtime_source_archive_sha256 -or
        $manifest.acceptance_artifacts.runtime_source_manifest.sha256 -ne $manifest.runtime.runtime_source_manifest_sha256 -or
        $manifest.runtime.gcp_image_self_link -ne $PinnedImage -or
        $manifest.runtime.gcp_image_id -ne $PinnedImageId -or
        $manifest.runtime.runtime_requirements_sha256 -ne $RuntimeRequirementsSha256 -or
        $manifest.runtime.runtime_fingerprint_sha256 -ne $RuntimeFingerprintSha256 -or
        $manifest.runtime.source_model_manifest_sha256 -ne $PinnedModelManifestSha256 -or
        $manifest.runtime.source_native_manifest_sha256 -ne $PinnedNativeManifestSha256 -or
        $manifest.launch_preflight.schema -ne $PreflightSchema -or
        $manifest.launch_preflight.status -ne "pass" -or
        [int]$manifest.launch_preflight.teacher_overlap_count -ne 0 -or
        [int]$manifest.launch_preflight.prior_population_overlap_count -ne 0) {
        throw "Frozen Attempt10 population run identity mismatch"
    }
    if ($packageReady.schema -ne $PackageReadySchema -or
        $packageReady.status -ne "immutable_package_only_complete" -or
        $packageReady.run_name -ne $RunName -or
        $packageReady.run_manifest_sha256 -ne (Get-Sha256 $manifestPath) -or
        $packageReady.source_sha256 -ne (Get-Sha256 $sourcePath) -or
        $packageReady.runtime_source_manifest_sha256 -ne $manifest.runtime.runtime_source_manifest_sha256 -or
        $packageReady.source_model_manifest_sha256 -ne $PinnedModelManifestSha256 -or
        $packageReady.source_native_manifest_sha256 -ne $PinnedNativeManifestSha256 -or
        $packageReady.runtime_dependency_closure_sha256 -ne $manifest.runtime.runtime_dependency_closure_sha256 -or
        $packageReady.current_profile_mutated -ne $false -or
        $packageReady.no_runtime_activation -ne $true) {
        throw "Attempt10 PACKAGE_READY authorization is missing, stale, or not bound to this immutable package"
    }
    foreach ($entry in @(
        @($sourcePath, $manifest.source.sha256),
        @($shardsPath, $manifest.shards.sha256),
        @($startupPath, $manifest.startup.sha256),
        @((Join-Path $packageDir "artifacts/m43_model.pkl"), $manifest.acceptance_artifacts.model.sha256),
        @((Join-Path $packageDir "artifacts/training_manifest.json"), $manifest.acceptance_artifacts.training_manifest.sha256),
        @((Join-Path $packageDir "artifacts/runtime_freeze.json"), $manifest.acceptance_artifacts.runtime_freeze.sha256),
        @((Join-Path $packageDir "artifacts/runtime_source.zip"), $manifest.acceptance_artifacts.runtime_source_archive.sha256),
        @((Join-Path $packageDir "artifacts/runtime_source_manifest.json"), $manifest.acceptance_artifacts.runtime_source_manifest.sha256),
        @((Join-Path $packageDir "artifacts/population_plan.json"), $manifest.population_plan.sha256)
    )) {
        Assert-Sha256Text $entry[1] "frozen manifest hash"
        if ((Get-Sha256 $entry[0]) -ne $entry[1]) { throw "Frozen Attempt10 local hash chain is broken" }
    }
    $oldPythonPath = $env:PYTHONPATH
    $oldDontWriteBytecode = $env:PYTHONDONTWRITEBYTECODE
    $env:PYTHONPATH = Join-Path $packageDir "runtime_source/src"
    $env:PYTHONDONTWRITEBYTECODE = "1"
    try {
        & python -c "import pathlib,sys; from ofc_regular.hu_m43_attempt10_distilled_runtime import validate_distilled_runtime_dependencies,validate_frozen_execution_modules; root=pathlib.Path(sys.argv[1]); validate_frozen_execution_modules(extracted_root=root/'runtime_source',manifest=root/'artifacts/runtime_source_manifest.json',module_names=['ofc_regular.hu_m43_attempt10_distilled_runtime']); validate_distilled_runtime_dependencies(root)" $packageDir
        if ($LASTEXITCODE -ne 0) { throw "Frozen Attempt10 resume dependency/source verification failed" }
    }
    finally {
        $env:PYTHONPATH = $oldPythonPath
        $env:PYTHONDONTWRITEBYTECODE = $oldDontWriteBytecode
    }
    # PackageOnly deliberately freezes local bytes before anything is published.
    # A resume may therefore see no remote manifest yet.  If it exists, it must
    # be byte-identical; the immutable upload below fills only absent objects.
    if (-not $DryRun -and (Test-GcsObject $manifestUri)) {
        $remoteManifest = [System.IO.Path]::GetTempFileName()
        try {
            Invoke-Gcloud @("storage", "cp", $manifestUri, $remoteManifest, "--project", $ProjectId) | Out-Null
            if ((Get-Sha256 $remoteManifest) -ne (Get-Sha256 $manifestPath)) {
                throw "Frozen remote Attempt10 manifest disagrees with the local resume manifest"
            }
        }
        finally {
            Remove-Item -LiteralPath $remoteManifest -Force -ErrorAction SilentlyContinue
        }
    }
}
else {
    if (Test-Path -LiteralPath $runDir) { throw "Attempt10 population run directory already exists and is immutable: $runDir" }
    $resolved = [ordered]@{
        model = Resolve-RepoPath $ModelPath
        training_manifest = Resolve-RepoPath $TrainingManifestPath
        runtime_freeze = Resolve-RepoPath $RuntimeFreezePath
        population_plan = Resolve-RepoPath $PlanPath
    }
    $resolvedRuntimeSourceArchive = Resolve-RepoPath $RuntimeSourceArchivePath
    $resolvedRuntimeSourceManifest = Resolve-RepoPath $RuntimeSourceManifestPath
    if (-not $FrozenRuntimeArtifactRootPath) {
        $FrozenRuntimeArtifactRootPath = "outputs/gcp_runs/$PinnedRuntimeTemplateRun/package_src"
    }
    $resolvedFrozenRuntimeArtifactRoot = Resolve-RepoPath $FrozenRuntimeArtifactRootPath
    New-Item -ItemType Directory -Path $packageDir -Force | Out-Null
    $runtimeSourceRoot = Join-Path $packageDir "runtime_source"
    Expand-Archive -LiteralPath $resolvedRuntimeSourceArchive -DestinationPath $runtimeSourceRoot
    $frozenPopulationPlan = Join-Path $runtimeSourceRoot "configs/hu_joint_policy_m43_attempt10_population.json"
    if ((Get-Sha256 $resolved.population_plan) -ne (Get-Sha256 $frozenPopulationPlan)) {
        throw "Requested Attempt10 population plan differs from the frozen runtime registry"
    }
    $resolved.population_plan = $frozenPopulationPlan
    $preflightArgs = @(
        "-m", "ofc_regular.validate_hu_m43_attempt10_acceptance", "preflight",
        "--model", $resolved.model,
        "--training-manifest", $resolved.training_manifest,
        "--runtime-freeze", $resolved.runtime_freeze,
        "--population-plan", $resolved.population_plan,
        "--runtime-source-archive", $resolvedRuntimeSourceArchive,
        "--runtime-source-manifest", $resolvedRuntimeSourceManifest,
        "--runtime-source-root", $runtimeSourceRoot,
        "--runtime-dependency-root", $resolvedFrozenRuntimeArtifactRoot
    )
    $oldPythonPath = $env:PYTHONPATH
    $oldDontWriteBytecode = $env:PYTHONDONTWRITEBYTECODE
    $oldPreference = $ErrorActionPreference
    $env:PYTHONPATH = Join-Path $runtimeSourceRoot "src"
    $env:PYTHONDONTWRITEBYTECODE = "1"
    $ErrorActionPreference = "Continue"
    try { $preflightRaw = @(& python @preflightArgs 2>&1); $preflightCode = $LASTEXITCODE }
    finally {
        $env:PYTHONPATH = $oldPythonPath
        $env:PYTHONDONTWRITEBYTECODE = $oldDontWriteBytecode
        $ErrorActionPreference = $oldPreference
    }
    if ($preflightCode -ne 0) { throw "Attempt10 population preflight failed ($preflightCode):`n$(@($preflightRaw) -join [Environment]::NewLine)" }
    $preflight = (@($preflightRaw) -join "`n") | ConvertFrom-Json
    if ($preflight.schema -ne $PreflightSchema -or $preflight.status -ne "pass" -or
        $preflight.model_schema -ne $ModelSchema -or $preflight.action_score_mode -ne $ActionScoreMode -or
        $preflight.artifact_schema -ne $ArtifactSchema -or
        $preflight.feature_schema -ne $FeatureSchema -or
        $preflight.head_schema -ne $HeadSchema -or
        $preflight.baseline_profile -ne $BaselineProfile -or
        $preflight.runtime_binding_verified -ne $true -or
        $preflight.teacher_calibration_locked_content_packaged -ne $false -or
        $preflight.runtime_requirements_sha256 -ne $RuntimeRequirementsSha256 -or
        $preflight.runtime_fingerprint_sha256 -ne $RuntimeFingerprintSha256 -or
        $preflight.source_model_manifest_sha256 -ne $PinnedModelManifestSha256 -or
        $preflight.source_native_manifest_sha256 -ne $PinnedNativeManifestSha256 -or
        [int]$preflight.teacher_overlap_count -ne 0 -or [int]$preflight.prior_population_overlap_count -ne 0) {
        throw "Attempt10 population preflight returned an unsafe receipt"
    }
    $plan = Get-Content -LiteralPath $resolved.population_plan -Raw | ConvertFrom-Json
    if ($plan.schema -ne "hu_m43_population_acceptance_plan_v1" -or
        $plan.status -ne "frozen_before_population_evaluation" -or
        $plan.fixed_baseline_profile -ne $BaselineProfile -or
        $plan.paired_seat_swap -ne $true -or
        [int]$plan.paired_seeds_per_opponent -ne 1000 -or
        [int]$plan.shards -ne 20 -or [int]$plan.paired_seeds_per_shard -ne 50 -or
        (@($plan.opponents) -join ",") -cne "stage19_p0,stage9f_p2,stage7_m5_r10,random_exact_final" -or
        $plan.activation_guards.current_profile_changed -ne $false -or
        $plan.activation_guards.runtime_policy_activated -ne $false -or
        $plan.post_acceptance_activation.population_complete_go_required -ne $true -or
        $plan.post_acceptance_activation.activation_mode_if_go -ne "explicit_opt_in_only" -or
        $plan.post_acceptance_activation.automatic_activation_allowed -ne $false -or
        $plan.post_acceptance_activation.current_profile_change_allowed -ne $false) {
        throw "Attempt10 population plan is not the frozen 1000x4, 20x50, stage19_p0-baseline contract"
    }

    # Source every baseline/native dependency from the exact manifest-pinned
    # development package closure.  Live repo models/target are never inputs.
    $dependencyRows = @()
    $sourceModelManifestPath = Join-Path $resolvedFrozenRuntimeArtifactRoot "source_model_manifest.json"
    $sourceNativeManifestPath = Join-Path $resolvedFrozenRuntimeArtifactRoot "source_native_manifest.json"
    if ((Get-Sha256 $sourceModelManifestPath) -ne $PinnedModelManifestSha256 -or
        (Get-Sha256 $sourceNativeManifestPath) -ne $PinnedNativeManifestSha256) {
        throw "Frozen development dependency manifest hash changed"
    }
    $sourceModelManifest = Get-Content -LiteralPath $sourceModelManifestPath -Raw | ConvertFrom-Json
    $sourceNativeManifest = Get-Content -LiteralPath $sourceNativeManifestPath -Raw | ConvertFrom-Json
    if ($sourceModelManifest.schema -ne "hu_m43_attempt02_source_model_manifest_v1" -or
        [int]$sourceModelManifest.model_count -ne 11 -or @($sourceModelManifest.models).Count -ne 11 -or
        $sourceNativeManifest.schema -ne "hu_m43_attempt02_source_native_manifest_v1" -or
        [int]$sourceNativeManifest.binary_count -ne 2 -or @($sourceNativeManifest.binaries).Count -ne 2) {
        throw "Frozen development dependency manifest semantics changed"
    }
    foreach ($item in @($sourceModelManifest.models) + @($sourceNativeManifest.binaries)) {
        $relative = [string]$item.path
        if ([System.IO.Path]::IsPathRooted($relative) -or $relative -match '(^|/|\\)\.\.(/|\\|$)' -or $relative -match '\\') {
            throw "Frozen dependency manifest contains an unsafe path: $relative"
        }
        $source = Join-Path $resolvedFrozenRuntimeArtifactRoot $relative
        if (-not (Test-Path -LiteralPath $source -PathType Leaf) -or
            [long](Get-Item -LiteralPath $source).Length -ne [long]$item.bytes -or
            (Get-Sha256 $source) -ne [string]$item.sha256) {
            throw "Frozen dependency bytes changed: $relative"
        }
        if ($relative -like "target/*") { Assert-LinuxX8664Elf $source }
        $destination = Join-Path $packageDir $relative
        New-Item -ItemType Directory -Path (Split-Path $destination -Parent) -Force | Out-Null
        Copy-Item -LiteralPath $source -Destination $destination
        $dependencyRows += [ordered]@{ path = $relative; bytes = [long]$item.bytes; sha256 = [string]$item.sha256 }
    }
    Copy-Item -LiteralPath $sourceModelManifestPath -Destination (Join-Path $packageDir "source_model_manifest.json")
    Copy-Item -LiteralPath $sourceNativeManifestPath -Destination (Join-Path $packageDir "source_native_manifest.json")
    $artifactDir = Join-Path $packageDir "artifacts"
    New-Item -ItemType Directory -Path $artifactDir -Force | Out-Null
    foreach ($name in $resolved.Keys) {
        $destinationName = if ($name -eq "model") { "m43_model.pkl" } else { "$name.json" }
        Copy-Item -LiteralPath $resolved[$name] -Destination (Join-Path $artifactDir $destinationName)
    }
    Copy-Item -LiteralPath $resolvedRuntimeSourceArchive -Destination (Join-Path $artifactDir "runtime_source.zip")
    Copy-Item -LiteralPath $resolvedRuntimeSourceManifest -Destination (Join-Path $artifactDir "runtime_source_manifest.json")
    if (Get-ChildItem -LiteralPath $artifactDir -Recurse -File | Where-Object { $_.Extension -eq ".jsonl" }) {
        throw "Teacher/calibration/locked JSONL must not enter the Attempt10 population package"
    }
    $artifactHashes = [ordered]@{
        model = Get-Sha256 (Join-Path $artifactDir "m43_model.pkl")
        training_manifest = Get-Sha256 (Join-Path $artifactDir "training_manifest.json")
        runtime_freeze = Get-Sha256 (Join-Path $artifactDir "runtime_freeze.json")
        runtime_source_archive = Get-Sha256 (Join-Path $artifactDir "runtime_source.zip")
        runtime_source_manifest = Get-Sha256 (Join-Path $artifactDir "runtime_source_manifest.json")
        population_plan = Get-Sha256 (Join-Path $artifactDir "population_plan.json")
    }
    foreach ($binding in @(
        @("model", "model_sha256"), @("training_manifest", "training_manifest_sha256"),
        @("runtime_freeze", "runtime_freeze_sha256"),
        @("runtime_source_archive", "runtime_source_archive_sha256"),
        @("runtime_source_manifest", "runtime_source_manifest_sha256"),
        @("population_plan", "population_plan_file_sha256")
    )) {
        if ($artifactHashes[$binding[0]] -ne $preflight.($binding[1])) { throw "Attempt10 lifecycle artifact changed during packaging: $($binding[0])" }
    }

    $shards = @()
    for ($index = 0; $index -lt [int]$plan.shards; $index++) {
        $offset = $index * [int]$plan.paired_seeds_per_shard
        $shards += [ordered]@{
            shard = $index
            offset = $offset
            seed = [long]$plan.seed + [long]$offset * [long]$plan.seed_stride
            seed_stride = [long]$plan.seed_stride
            paired_seeds = [int]$plan.paired_seeds_per_shard
            output_prefix = ("shard-{0:D4}" -f $index)
        }
    }
    Write-Utf8NoBom $shardsPath ((@($shards | ForEach-Object { $_ | ConvertTo-Json -Compress }) -join "`n") + "`n")
    Copy-Item -LiteralPath $shardsPath -Destination (Join-Path $packageDir "population_shards.jsonl")
    Write-Utf8NoBom (Join-Path $packageDir "population_source_models.json") (($dependencyRows | ConvertTo-Json -Depth 5) + "`n")
    if (Test-Path -LiteralPath (Join-Path $packageDir "configs")) {
        throw "Live repository configs/current-profile metadata must not enter the Attempt10 population package"
    }
    $expectedRuntimeConfigs = @(
        "hu_m43_attempt08_runtime_requirements.txt",
        "hu_joint_policy_m43_population.json",
        "hu_joint_policy_m43_attempt03_population.json",
        "hu_joint_policy_m43_attempt05.json",
        "hu_joint_policy_m43_attempt06.json",
        "hu_joint_policy_m43_attempt07.json",
        "hu_joint_policy_m43_attempt07_preflight.json",
        "hu_joint_policy_m43_attempt10.json",
        "hu_joint_policy_m43_attempt10_preflight.json",
        "hu_joint_policy_m43_attempt10_population.json"
    ) | Sort-Object
    $runtimeConfigs = @(Get-ChildItem -LiteralPath (Join-Path $runtimeSourceRoot "configs") -File | ForEach-Object Name | Sort-Object)
    if ((@($runtimeConfigs) -join ",") -cne (@($expectedRuntimeConfigs) -join ",")) {
        throw "Frozen runtime source registry/requirements config set changed"
    }
    if (Test-Path -LiteralPath (Join-Path $packageDir "rust")) {
        throw "Rust source/debug artifacts must not enter the Attempt10 population package"
    }
    $allowedJsonl = [System.IO.Path]::GetFullPath((Join-Path $packageDir "population_shards.jsonl"))
    $unexpectedJsonl = @(Get-ChildItem -LiteralPath $packageDir -Recurse -File | Where-Object {
        $_.Extension -ieq ".jsonl" -and $_.FullName -cne $allowedJsonl
    })
    if ($unexpectedJsonl.Count) {
        throw "Teacher/calibration/locked JSONL found outside the sole shard-spec allowlist entry"
    }
    $oldPythonPath = $env:PYTHONPATH
    $oldDontWriteBytecode = $env:PYTHONDONTWRITEBYTECODE
    $env:PYTHONPATH = Join-Path $runtimeSourceRoot "src"
    $env:PYTHONDONTWRITEBYTECODE = "1"
    try {
        & python -c "import pathlib,sys; from ofc_regular.hu_m43_attempt10_distilled_runtime import validate_distilled_runtime_dependencies,validate_frozen_execution_modules; root=pathlib.Path(sys.argv[1]); validate_frozen_execution_modules(extracted_root=root/'runtime_source',manifest=root/'artifacts/runtime_source_manifest.json',module_names=['ofc_regular.hu_m43_attempt10_distilled_runtime']); validate_distilled_runtime_dependencies(root)" $packageDir
        if ($LASTEXITCODE -ne 0) { throw "Packaged Attempt10 dependency/source verification failed" }
    }
    finally {
        $env:PYTHONPATH = $oldPythonPath
        $env:PYTHONDONTWRITEBYTECODE = $oldDontWriteBytecode
    }
    New-ZipWithForwardSlashes $packageDir $sourcePath

    $startup = @'
#!/usr/bin/env bash
set -Eeuo pipefail
LOG=/var/log/hu_m43_attempt10_population.log
exec > >(tee -a "$LOG") 2>&1
meta(){ curl -fsS -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }
RUN_NAME="$(meta RUN_NAME)"; BUCKET="$(meta BUCKET)"; SHARD="$(meta SHARD)"
SOURCE_URI="$(meta SOURCE_URI)"; SOURCE_SHA="$(meta SOURCE_SHA)"; MANIFEST_SHA="$(meta MANIFEST_SHA)"
SEED_REGISTRY_SHA="$(meta SEED_REGISTRY_SHA)"
SYNC_SECONDS="$(meta SYNC_SECONDS)"; SELF_DELETE="$(meta SELF_DELETE)"
INSTANCE="$(curl -fsS -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/name)"
ZONE_URL="$(curl -fsS -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/zone)"; ZONE="${ZONE_URL##*/}"
PREFIX="gs://${BUCKET}/runs/${RUN_NAME}"; OUT=/work/out; WORK=/work/repo; mkdir -p "$OUT"
STATUS="$OUT/status.json"; HEARTBEAT_PID=""
write_status(){ python3 - "$STATUS" "$RUN_NAME" "$SHARD" "$1" "$2" "$INSTANCE" "$ZONE" "$MANIFEST_SHA" <<'PY'
import datetime,json,sys
p,run,shard,state,code,instance,zone,manifest=sys.argv[1:]
json.dump({'schema':'hu_m43_attempt10_population_spot_status_v1','run_name':run,'shard':int(shard),'state':state,'exit_code':int(code),'instance':instance,'zone':zone,'manifest_sha256':manifest,'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat()},open(p,'w'),sort_keys=True)
PY
gcloud storage cp "$STATUS" "$PREFIX/status/shard-${SHARD}.json" >/dev/null || true; }
cleanup(){ code=$?; [[ -z "$HEARTBEAT_PID" ]] || kill "$HEARTBEAT_PID" >/dev/null 2>&1 || true; if [[ $code -ne 0 ]]; then write_status failed "$code"; gcloud storage cp "$LOG" "$PREFIX/results/shard-$(printf '%04d' "$SHARD")/startup.log" >/dev/null || true; fi; if [[ "$SELF_DELETE" == 1 ]]; then gcloud compute instances delete "$INSTANCE" --zone "$ZONE" --quiet >/dev/null 2>&1 || true; fi; exit "$code"; }
trap cleanup EXIT
write_status booting 0
RESULT="$PREFIX/results/shard-$(printf '%04d' "$SHARD")"
if gcloud storage ls "$RESULT/DONE" >/dev/null 2>&1; then write_status complete 0; exit 0; fi
export DEBIAN_FRONTEND=noninteractive
apt-get update -y; apt-get install -y unzip curl ca-certificates python3 python3-venv python3-pip
if ! command -v gcloud >/dev/null 2>&1; then
  apt-get install -y apt-transport-https gnupg
  curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
  echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" >/etc/apt/sources.list.d/google-cloud-sdk.list
  apt-get update -y; apt-get install -y google-cloud-cli
fi
gcloud storage cp "$SOURCE_URI" /tmp/source.zip >/dev/null
echo "${SOURCE_SHA}  /tmp/source.zip" | sha256sum -c -
rm -rf "$WORK"; mkdir -p "$WORK"; unzip -q /tmp/source.zip -d "$WORK"; cd "$WORK"
SPEC="$(sed -n "$((SHARD+1))p" population_shards.jsonl)"; [[ -n "$SPEC" ]]
eval "$(python3 - "$SPEC" <<'PY'
import json,shlex,sys
x=json.loads(sys.argv[1])
for k,v in {'SEED':x['seed'],'STRIDE':x['seed_stride'],'COUNT':x['paired_seeds']}.items(): print(f'{k}={shlex.quote(str(v))}')
PY
)"
python3 -m venv .venv; source .venv/bin/activate
python -m pip install -r runtime_source/configs/hu_m43_attempt08_runtime_requirements.txt
export PYTHONPATH="$WORK/runtime_source/src" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 OFC_HU_M3_BATCH_THREADS=4
python - <<'PY'
from ofc_regular.hu_m43_attempt10_distilled_runtime import validate_distilled_runtime_dependencies, validate_distilled_runtime_extracted_tree, validate_distilled_runtime_source_archive, validate_frozen_execution_modules
from ofc_regular.hu_m43_attempt08_runtime_identity import validate_expected_runtime_fingerprint
from ofc_regular.hu_m3_rust import engine_version, load_native_engine
from ofc_regular.hu_turn3_stage3_feature_rust import rust_direct_available
validate_distilled_runtime_source_archive(archive_path='artifacts/runtime_source.zip', manifest='artifacts/runtime_source_manifest.json')
validate_distilled_runtime_extracted_tree(extracted_root='runtime_source', manifest='artifacts/runtime_source_manifest.json')
validate_frozen_execution_modules(extracted_root='runtime_source', manifest='artifacts/runtime_source_manifest.json', module_names=['ofc_regular.hu_m43_attempt10_distilled_runtime'])
validate_distilled_runtime_dependencies('.')
assert validate_expected_runtime_fingerprint() == '8c2cd111bc4e70096ff4f974f684ad146e94329871328e5b5db5d3426256c218'
print('hu_m3_engine='+engine_version(library=load_native_engine()))
assert rust_direct_available(), 'Stage3 native feature encoder unavailable'
PY
MODEL_SHA="$(sha256sum artifacts/m43_model.pkl | cut -d' ' -f1)"
write_status evaluating 0
(while true; do sleep "$SYNC_SECONDS"; write_status evaluating 0; done) & HEARTBEAT_PID=$!
python -m ofc_regular.evaluate_hu_m4_population --model artifacts/m43_model.pkl --expected-model-sha256 "$MODEL_SHA" --freeze-manifest artifacts/runtime_freeze.json --training-manifest artifacts/training_manifest.json --runtime-source-manifest artifacts/runtime_source_manifest.json --runtime-source-root runtime_source --runtime-dependency-root . --seed-registry-sha256 "$SEED_REGISTRY_SHA" --baseline-profile stage19_p0 --paired-seeds "$COUNT" --seed "$SEED" --seed-stride "$STRIDE" --opponents stage19_p0 stage9f_p2 stage7_m5_r10 random_exact_final --records-output "$OUT/records.jsonl" --output "$OUT/evaluation.json" --progress-every 10 > "$OUT/evaluation_stdout.log"
kill "$HEARTBEAT_PID" >/dev/null 2>&1 || true; HEARTBEAT_PID=""
python - "$OUT/evaluation.json" "$OUT/records.jsonl" "$SEED" "$STRIDE" "$COUNT" "$MODEL_SHA" <<'PY'
import json,sys
e=json.load(open(sys.argv[1])); rows=[json.loads(x) for x in open(sys.argv[2]) if x.strip()]
seed,stride,count=int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5]); model=sys.argv[6]
assert e['schema']=='hu_m4_t1_population_evaluation_v1' and e['paired_seat_swap'] is True
assert e['seed']==seed and e['seed_stride']==stride and e['paired_seeds_per_opponent']==count
assert e['opponents']==['stage19_p0','stage9f_p2','stage7_m5_r10','random_exact_final']
assert e['runtime_config']['baseline_profile']=='stage19_p0'
assert len(rows)==count*8 and e['trace_hands']==count*16
assert all(row.get('runtime_binding_verified') is True for row in rows)
r=e['runtime_config']
assert r['current_profile_used'] is False and r['promotion_artifact_contract'] is True and r['diagnostic_legacy'] is False
assert r['candidate_model_sha256']==model and r['safety_model_sha256']==model
assert r['model_schema']=='hu_m43_attempt10_t1_second_distilled_selector_v1' and r['runtime_binding_verified'] is True
assert r['artifact_schema']=='hu_m43_attempt10_t1_second_distilled_pickle_v1'
assert r['feature_schema']=='hu_m43_attempt10_lambda_top12_public_infoset_features_v1'
assert r['head_schema']=='hu_m43_attempt10_policy_delta_safe_tail_heads_v1'
assert r['action_score_mode']=='attempt10_lambda_top12_distilled_safe_selector_v1' and r['safety_enabled'] is True
assert r['source_model_manifest_sha256']=='e9b9d9748bb2b2f31a4baa0d928b60c05f254ab5d7915a7461fb8402e1e7ddb8'
assert r['source_native_manifest_sha256']=='ec295d070ac7bb21a82aeb2f432b8f6f191e61b63027bc446e8bd5f39f51569f'
assert len(r['runtime_dependency_closure_sha256'])==64
PY
EVAL_SHA="$(sha256sum "$OUT/evaluation.json" | cut -d' ' -f1)"; RECORDS_SHA="$(sha256sum "$OUT/records.jsonl" | cut -d' ' -f1)"
python - "$OUT/DONE" "$RUN_NAME" "$SHARD" "$MANIFEST_SHA" "$SOURCE_SHA" "$MODEL_SHA" "$EVAL_SHA" "$RECORDS_SHA" <<'PY'
import json,sys
p,run,shard,manifest,source,model,evaluation,records=sys.argv[1:]
json.dump({'schema':'hu_m43_attempt10_population_spot_done_v1','status':'complete','run_name':run,'shard':int(shard),'manifest_sha256':manifest,'source_sha256':source,'model_sha256':model,'evaluation_sha256':evaluation,'records_sha256':records,'current_profile_mutated':False,'no_runtime_activation':True},open(p,'w'),sort_keys=True)
PY
gcloud storage cp "$OUT/evaluation.json" "$RESULT/evaluation.json" >/dev/null
gcloud storage cp "$OUT/records.jsonl" "$RESULT/records.jsonl" >/dev/null
gcloud storage cp "$OUT/evaluation_stdout.log" "$RESULT/evaluation_stdout.log" >/dev/null
gcloud storage cp "$LOG" "$RESULT/startup.log" >/dev/null
write_status complete 0
gcloud storage cp "$OUT/DONE" "$RESULT/DONE" --if-generation-match=0 >/dev/null
'@
    Write-Utf8NoBom $startupPath ($startup + "`n")

    $manifest = [ordered]@{
        schema = $ManifestSchema
        run_name = $RunName
        project_id = $ProjectId
        bucket = $Bucket
        population_plan = [ordered]@{ uri = $planUri; sha256 = [string]$preflight.population_plan_file_sha256; paired_seeds = [int]$plan.paired_seeds_per_opponent; seed = [long]$plan.seed; seed_stride = [long]$plan.seed_stride; shards = [int]$plan.shards; paired_seeds_per_shard = [int]$plan.paired_seeds_per_shard }
        acceptance_artifacts = [ordered]@{
            model = [ordered]@{ uri = $modelUri; sha256 = [string]$preflight.model_sha256 }
            training_manifest = [ordered]@{ uri = $trainingManifestUri; sha256 = [string]$preflight.training_manifest_sha256 }
            runtime_freeze = [ordered]@{ uri = $runtimeFreezeUri; sha256 = [string]$preflight.runtime_freeze_sha256 }
            runtime_source_archive = [ordered]@{ uri = $runtimeSourceArchiveUri; sha256 = [string]$preflight.runtime_source_archive_sha256 }
            runtime_source_manifest = [ordered]@{ uri = $runtimeSourceManifestUri; sha256 = [string]$preflight.runtime_source_manifest_sha256 }
        }
        source = [ordered]@{ uri = $sourceUri; sha256 = Get-Sha256 $sourcePath; bytes = [long](Get-Item $sourcePath).Length }
        startup = [ordered]@{ uri = $startupUri; sha256 = Get-Sha256 $startupPath }
        shards = [ordered]@{ uri = $shardsUri; sha256 = Get-Sha256 $shardsPath; count = [int]$plan.shards }
        runtime = [ordered]@{ model_sha256 = [string]$preflight.model_sha256; model_schema = $ModelSchema; artifact_schema = $ArtifactSchema; feature_schema = $FeatureSchema; head_schema = $HeadSchema; model_id = [string]$preflight.model_id; action_score_mode = $ActionScoreMode; baseline_profile = $BaselineProfile; fixed_safe_probability_threshold = [double]$preflight.fixed_safe_probability_threshold; fixed_fold_votes_min = [int]$preflight.fixed_fold_votes_min; training_manifest_sha256 = [string]$preflight.training_manifest_sha256; runtime_freeze_sha256 = [string]$preflight.runtime_freeze_sha256; runtime_source_archive_sha256 = [string]$preflight.runtime_source_archive_sha256; runtime_source_manifest_sha256 = [string]$preflight.runtime_source_manifest_sha256; runtime_source_closure_sha256 = [string]$preflight.runtime_source_closure_sha256; runtime_semantic_closure_sha256 = [string]$preflight.runtime_semantic_closure_sha256; source_model_manifest_sha256 = [string]$preflight.source_model_manifest_sha256; source_native_manifest_sha256 = [string]$preflight.source_native_manifest_sha256; runtime_dependency_closure_sha256 = [string]$preflight.runtime_dependency_closure_sha256; runtime_requirements_sha256 = $RuntimeRequirementsSha256; runtime_fingerprint_sha256 = $RuntimeFingerprintSha256; gcp_image_self_link = $PinnedImage; gcp_image_id = $PinnedImageId; seed_registry_sha256 = [string]$preflight.seed_registry_sha256; runtime_teacher_inputs = $false; teacher_ev_lcb_runtime_gate = $false; current_profile_used = $false }
        launch_preflight = $preflight
        source_boundary = [ordered]@{ teacher_jsonl_packaged = $false; audit_jsonl_packaged = $false; calibration_jsonl_packaged = $false; locked_jsonl_packaged = $false; current_profile_artifact_packaged = $false; runtime_artifacts_only = $true }
        compute = [ordered]@{ machine_type = $MachineType; fallback_machine_types = $FallbackMachineTypes; zones = $Zones; boot_disk_gb = $BootDiskGb; provisioning_model = "SPOT"; instance_termination_action = "DELETE"; self_delete = (-not [bool]$NoSelfDelete); sync_interval_seconds = $SyncIntervalSeconds }
        checkpoint = [ordered]@{ unit = "completed_shard"; retry = "deterministic_full_shard"; resume_missing_shards_only = $true; done_commit_last = $true }
        no_runtime_activation = $true
        current_profile_mutated = $false
    }
    Write-Utf8NoBom $manifestPath (($manifest | ConvertTo-Json -Depth 15) + "`n")
}

$manifestSha = Get-Sha256 $manifestPath
if ($PackageOnly) {
    if (Test-Path -LiteralPath $packageReadyPath) { throw "Attempt10 PACKAGE_READY marker already exists" }
    $packageReady = [ordered]@{
        schema = $PackageReadySchema
        status = "immutable_package_only_complete"
        run_name = $RunName
        run_manifest_sha256 = $manifestSha
        source_sha256 = Get-Sha256 $sourcePath
        runtime_source_manifest_sha256 = [string]$manifest.runtime.runtime_source_manifest_sha256
        source_model_manifest_sha256 = $PinnedModelManifestSha256
        source_native_manifest_sha256 = $PinnedNativeManifestSha256
        runtime_dependency_closure_sha256 = [string]$manifest.runtime.runtime_dependency_closure_sha256
        current_profile_mutated = $false
        no_runtime_activation = $true
    }
    Write-NewUtf8NoBom $packageReadyPath (($packageReady | ConvertTo-Json -Depth 6) + "`n")
    [ordered]@{ schema = "hu_m43_attempt10_population_package_result_v1"; run_name = $RunName; mode = "package_only"; run_dir = $runDir; manifest_sha256 = $manifestSha; source_sha256 = Get-Sha256 $sourcePath; package_ready_sha256 = Get-Sha256 $packageReadyPath; shards = [int]$manifest.shards.count; no_runtime_activation = $true; current_profile_mutated = $false } | ConvertTo-Json -Depth 6
    exit 0
}
if ($DryRun) {
    [ordered]@{ schema = "hu_m43_attempt10_population_package_result_v1"; run_name = $RunName; mode = "dry_run"; run_dir = $runDir; manifest_sha256 = $manifestSha; source_sha256 = Get-Sha256 $sourcePath; shards = [int]$manifest.shards.count; no_runtime_activation = $true; current_profile_mutated = $false } | ConvertTo-Json -Depth 6
    exit 0
}

foreach ($pair in @(
    @($sourcePath, $sourceUri), @($startupPath, $startupUri),
    @($shardsPath, $shardsUri), @((Join-Path $packageDir "artifacts/population_plan.json"), $planUri),
    @((Join-Path $packageDir "artifacts/m43_model.pkl"), $modelUri),
    @((Join-Path $packageDir "artifacts/training_manifest.json"), $trainingManifestUri),
    @((Join-Path $packageDir "artifacts/runtime_freeze.json"), $runtimeFreezeUri),
    @((Join-Path $packageDir "artifacts/runtime_source.zip"), $runtimeSourceArchiveUri),
    @((Join-Path $packageDir "artifacts/runtime_source_manifest.json"), $runtimeSourceManifestUri)
)) {
    Publish-ImmutableObject $pair[0] $pair[1]
}
Publish-ImmutableObject $manifestPath $manifestUri
if (-not $CreateInstances) {
    [ordered]@{ run_name = $RunName; state = "packaged_and_uploaded"; manifest_sha256 = $manifestSha; create_instances = $false; no_runtime_activation = $true } | ConvertTo-Json -Depth 6
    exit 0
}

$selected = if ($StartShards.Count) { @($StartShards | ForEach-Object { [int]$_ }) } else { @(0..([int]$manifest.shards.count - 1)) }
if (@($selected | Select-Object -Unique).Count -ne $selected.Count) {
    throw "Attempt10 population shard selection contains duplicates"
}
if (@($selected | Where-Object { $_ -ne 0 }).Count) {
    $canaryUri = "$gcsPrefix/results/shard-0000/DONE"
    if (-not (Test-GcsObject $canaryUri)) {
        throw "Attempt10 population fanout requires a completed shard-0000 canary"
    }
    $canary = ((Invoke-Gcloud @("storage", "cat", $canaryUri, "--project", $ProjectId)) -join "`n") | ConvertFrom-Json
    if ($canary.schema -ne $DoneSchema -or $canary.status -ne "complete" -or
        $canary.run_name -ne $RunName -or [int]$canary.shard -ne 0 -or
        $canary.manifest_sha256 -ne $manifestSha -or
        $canary.source_sha256 -ne $manifest.source.sha256 -or
        $canary.model_sha256 -ne $manifest.runtime.model_sha256 -or
        $canary.current_profile_mutated -ne $false -or $canary.no_runtime_activation -ne $true) {
        throw "Attempt10 population fanout canary is not bound to the frozen run"
    }
}
foreach ($shard in $selected) {
    if ($shard -lt 0 -or $shard -ge [int]$manifest.shards.count) { throw "Shard index outside frozen Attempt10 plan: $shard" }
    $resultPrefix = "shard-{0:D4}" -f $shard
    if (Test-GcsObject "$gcsPrefix/results/$resultPrefix/DONE") { continue }
    $vmName = "$vmPrefix-s{0:D3}" -f $shard
    $existing = Invoke-Gcloud @("compute", "instances", "list", "--project", $ProjectId, "--filter", "name=$vmName", "--format=value(name)")
    if (@($existing | Where-Object { $_ }).Count) {
        if ($SkipExistingInstances) { continue }
        throw "Attempt10 population worker already exists: $vmName"
    }
    $created = $false; $errors = @(); $attempt = 0
    foreach ($machine in @($MachineType) + @($FallbackMachineTypes)) {
        $zone = $Zones[$attempt % $Zones.Count]; $attempt++
        $arguments = @("compute", "instances", "create", $vmName, "--project", $ProjectId, "--zone", $zone, "--machine-type", $machine, "--provisioning-model", "SPOT", "--instance-termination-action", "DELETE", "--boot-disk-size", "${BootDiskGb}GB", "--image", $PinnedImage, "--scopes", "cloud-platform", "--metadata", "RUN_NAME=$RunName,BUCKET=$Bucket,SHARD=$shard,SOURCE_URI=$($manifest.source.uri),SOURCE_SHA=$($manifest.source.sha256),MANIFEST_SHA=$manifestSha,SEED_REGISTRY_SHA=$($manifest.runtime.seed_registry_sha256),SYNC_SECONDS=$SyncIntervalSeconds,SELF_DELETE=$(if ($NoSelfDelete) { 0 } else { 1 })", "--metadata-from-file", "startup-script=$startupPath")
        try { Invoke-Gcloud $arguments | Out-Null; $created = $true; break }
        catch { $errors += $_.Exception.Message }
    }
    if (-not $created) { throw "Could not create Attempt10 population worker $vmName`n$($errors -join [Environment]::NewLine)" }
}
[ordered]@{ run_name = $RunName; state = "workers_started"; shards = $selected; manifest_sha256 = $manifestSha; provisioning_model = "SPOT"; no_runtime_activation = $true; current_profile_mutated = $false } | ConvertTo-Json -Depth 8
