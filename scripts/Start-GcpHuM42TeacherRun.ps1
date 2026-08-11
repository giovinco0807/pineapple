param(
    [string]$ProjectId = "ofc-solver-485418",
    [string]$Bucket = "pokerhu-ofc-solver-485418-training",
    [string]$RunName = ("regular-hu-m42-c2e8-gate40-" + (Get-Date -Format "yyyyMMdd-HHmmss")),
    [int]$TrainRoots = 20,
    [int]$CalibrationRoots = 10,
    [int]$LockedHoldoutRoots = 10,
    [ValidateSet(5, 10)]
    [int]$RootsPerShard = 5,
    [long]$TrainSeedBase = 1706071901,
    [long]$CalibrationSeedBase = 1806071901,
    [long]$LockedHoldoutSeedBase = 1906071901,
    [long]$SeedStride = 1000003,
    [long]$TrainCandidateSeedBase = 1106071901,
    [long]$CalibrationCandidateSeedBase = 1206071901,
    [long]$LockedCandidateSeedBase = 1306071901,
    [long]$TrainEvaluationSeedBase = 1406071901,
    [long]$CalibrationEvaluationSeedBase = 1506071901,
    [long]$LockedEvaluationSeedBase = 1606071901,
    [long]$TrainChildPolicySeedBase = 2006071901,
    [long]$CalibrationChildPolicySeedBase = 2016071901,
    [long]$LockedChildPolicySeedBase = 2026071901,
    [int]$CandidateSamples = 2,
    [int]$EvaluationSamples = 8,
    [int]$NativeBatchThreads = 4,
    [int]$SyncIntervalSeconds = 60,
    [string]$MachineType = "c4-standard-4",
    [string]$BootDiskType = "hyperdisk-balanced",
    [string[]]$FallbackMachineTypes = @("n2-standard-4", "e2-standard-4"),
    [string]$FallbackBootDiskType = "pd-balanced",
    [string[]]$Zones = @(
        "asia-northeast1-b",
        "asia-northeast1-c",
        "us-central1-a"
    ),
    [int]$BootDiskGb = 50,
    [string[]]$InitialShards = @(),
    [string[]]$StartShards = @(),
    [switch]$AllowNativeSourceBuildFallback,
    [switch]$CreateInstances,
    [switch]$SkipExistingInstances,
    [switch]$NoSelfDelete,
    [switch]$PackageOnly,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$scriptBoundParameters = @{} + $PSBoundParameters

$RootProfiles = @(
    "stage19_p0",
    "stage9f_p2",
    "stage7_m5_r10",
    "stage3_baseline",
    "random_exact_final"
)
$RootProfileWeights = @(1.0, 1.0, 1.0, 1.0, 1.0)
$BaselineProfile = "stage18_p1"
$T2Profile = "stage9f_p2"

# Keep this list exact. These are the only model binaries copied into the
# dirty-worktree source archive used by an M4.2 worker.
$RequiredModels = @(
    "models/opening_stage7_torch_wide.pt",
    "models/turn1_stage6_torch_wide.pt",
    "models/turn2_stage8.pkl",
    "models/turn3_stage6.pkl",
    "models/hu_turn2_stage8b_safe_lcb196_20k_reference_override_cached_rank_wide.pt",
    "models/hu_turn3_stage7_reference_override_cached_rank_wide.pt",
    "models/hu_turn3_stage3_mc32_500k_plus_m8_12_f128_100k_w2_cached_rank_wide.pt",
    "models/hu_turn1_stage18_first1801_mc32_hgb_abs_aug3.pkl",
    "models/hu_turn1_stage18_highmc_safe_selector_a_meta_only.pkl",
    "models/hu_turn0_stage3_top60_mc32_1000_hgb_baseline-delta_sew1_aug3.pkl",
    "models/hu_turn0_stage19_safe_selector_highmc_candidate_delta_plus_meta.pkl"
)

# These are Linux x86_64 ELF libraries built locally in the pinned Docker
# builder. Workers load these exact files; they do not run Cargo by default.
$RequiredNativeBinaries = @(
    "target/release/libofc_stage3_feature_encoder.so",
    "target/release/libofc_hu_m3_engine.so"
)
$PinnedNativeArtifacts = @{
    "target/release/libofc_stage3_feature_encoder.so" = [pscustomobject]@{
        bytes = 500712
        sha256 = "9e58797bd234f9858fdfeee69fee03f6d0d20b6e3082de5ab54e9a6356ad0ea7"
    }
    "target/release/libofc_hu_m3_engine.so" = [pscustomobject]@{
        bytes = 1124776
        sha256 = "70523749b757a92b8547a5753460c0d9c3435b6872ca76c231a8c38ecc1bbc36"
    }
}

function Write-Utf8NoBom {
    param([string]$Path, [string]$Text)
    $encoding = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::WriteAllText($Path, $Text, $encoding)
}

function Get-Sha256 {
    param([string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Assert-LinuxX8664Elf {
    param([string]$Path)
    $stream = [System.IO.File]::OpenRead($Path)
    try {
        $header = New-Object byte[] 20
        if ($stream.Read($header, 0, $header.Length) -ne $header.Length -or
            $header[0] -ne 0x7f -or $header[1] -ne 0x45 -or
            $header[2] -ne 0x4c -or $header[3] -ne 0x46 -or
            $header[4] -ne 2 -or $header[5] -ne 1 -or
            $header[16] -ne 0x03 -or $header[17] -ne 0x00 -or
            $header[18] -ne 0x3e -or $header[19] -ne 0x00) {
            throw "Pinned native binary is not a Linux x86_64 ELF shared library: $Path"
        }
    }
    finally {
        $stream.Dispose()
    }
}

function Assert-PinnedNativeArtifact {
    param([string]$RelativePath, [string]$Path)
    $expected = $PinnedNativeArtifacts[$RelativePath]
    if ($null -eq $expected -or
        [long](Get-Item -LiteralPath $Path).Length -ne [long]$expected.bytes -or
        (Get-Sha256 $Path) -ne [string]$expected.sha256) {
        throw "Native binary does not match the pinned Docker artifact: $RelativePath"
    }
}

function Convert-ToVmPrefix {
    param([string]$Name)
    $value = ($Name.ToLowerInvariant() -replace '[^a-z0-9-]', '-').Trim('-')
    if (-not $value) { throw "RunName does not produce a valid VM name" }
    if ($value.Length -gt 54) { $value = $value.Substring(0, 54).Trim('-') }
    return $value
}

function New-ZipWithForwardSlashes {
    param([string]$SourceDir, [string]$DestinationPath)
    Add-Type -AssemblyName System.IO.Compression
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    if (Test-Path $DestinationPath) { Remove-Item -LiteralPath $DestinationPath -Force }
    $sourceRoot = (Resolve-Path -LiteralPath $SourceDir).Path.TrimEnd('\', '/')
    $prefixLength = $sourceRoot.Length + 1
    $zip = [System.IO.Compression.ZipFile]::Open(
        $DestinationPath,
        [System.IO.Compression.ZipArchiveMode]::Create
    )
    try {
        Get-ChildItem -LiteralPath $sourceRoot -Recurse -File | ForEach-Object {
            $entryName = $_.FullName.Substring($prefixLength) -replace '\\', '/'
            [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
                $zip,
                $_.FullName,
                $entryName,
                [System.IO.Compression.CompressionLevel]::Optimal
            ) | Out-Null
        }
    }
    finally {
        $zip.Dispose()
    }
}

if ($TrainRoots -le 0 -or $CalibrationRoots -le 0 -or $LockedHoldoutRoots -le 0) {
    throw "Every M4.2 split must contain at least one root"
}
foreach ($value in @($TrainRoots, $CalibrationRoots, $LockedHoldoutRoots)) {
    if (($value % $RootsPerShard) -ne 0) {
        throw "Each split root count must be divisible by RootsPerShard"
    }
}
if (($RootsPerShard % $RootProfiles.Count) -ne 0) {
    throw "RootsPerShard must preserve an equal five-profile quota"
}
if ($CandidateSamples -le 0 -or $EvaluationSamples -le 0) {
    throw "CandidateSamples and EvaluationSamples must be positive"
}
if ($NativeBatchThreads -lt 1 -or $NativeBatchThreads -gt 64) {
    throw "NativeBatchThreads must be between 1 and 64"
}
if ($SyncIntervalSeconds -lt 10) { throw "SyncIntervalSeconds must be at least 10" }
if ($BootDiskGb -lt 20) { throw "BootDiskGb must be at least 20" }
if (-not $Zones -or $Zones.Count -eq 0) { throw "At least one zone is required" }
if (-not $MachineType -or -not $BootDiskType) {
    throw "MachineType and BootDiskType must not be empty"
}
if ($RunName -notmatch '^[a-zA-Z0-9][a-zA-Z0-9._-]*$') {
    throw "RunName may only contain letters, numbers, dot, underscore, and dash"
}
if ($PackageOnly -and $CreateInstances) {
    throw "PackageOnly and CreateInstances are mutually exclusive"
}

$allSeedBases = @(
    $TrainSeedBase, $CalibrationSeedBase, $LockedHoldoutSeedBase,
    $TrainCandidateSeedBase, $CalibrationCandidateSeedBase, $LockedCandidateSeedBase,
    $TrainEvaluationSeedBase, $CalibrationEvaluationSeedBase, $LockedEvaluationSeedBase,
    $TrainChildPolicySeedBase, $CalibrationChildPolicySeedBase, $LockedChildPolicySeedBase
)
if (@($allSeedBases | Sort-Object -Unique).Count -ne $allSeedBases.Count) {
    throw "All split, candidate, evaluation, and child-policy seed bases must be distinct"
}
if ($TrainCandidateSeedBase -eq $TrainEvaluationSeedBase -or
    $CalibrationCandidateSeedBase -eq $CalibrationEvaluationSeedBase -or
    $LockedCandidateSeedBase -eq $LockedEvaluationSeedBase) {
    throw "Candidate and evaluation seeds must be independent in every split"
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$modelPlan = @()
if ($DryRun) {
    foreach ($model in $RequiredModels) {
        $modelPath = Join-Path $repoRoot $model
        if (-not (Test-Path -LiteralPath $modelPath -PathType Leaf)) {
            throw "Required M4.2 model is missing: $model"
        }
        $item = Get-Item -LiteralPath $modelPath
        $modelPlan += [pscustomobject]@{ path = $model; bytes = [long]$item.Length }
    }
}
if ($RequiredModels.Count -ne 11) { throw "M4.2 requires exactly 11 model files" }

$nativePlan = @()
if ($DryRun) {
    foreach ($native in $RequiredNativeBinaries) {
        $nativePath = Join-Path $repoRoot $native
        if (-not (Test-Path -LiteralPath $nativePath -PathType Leaf)) {
            throw "Required Docker-built Linux native binary is missing: $native"
        }
        Assert-LinuxX8664Elf -Path $nativePath
        Assert-PinnedNativeArtifact -RelativePath $native -Path $nativePath
        $nativeItem = Get-Item -LiteralPath $nativePath
        $nativePlan += [pscustomobject]@{
            path = $native
            bytes = [long]$nativeItem.Length
            sha256 = Get-Sha256 $nativePath
        }
    }
}
if ($RequiredNativeBinaries.Count -ne 2) { throw "M4.2 requires exactly two pinned native binaries" }

$shardSpecs = New-Object System.Collections.Generic.List[object]
function Add-SplitShards {
    param(
        [string]$Split,
        [int]$Roots,
        [long]$SeedBase,
        [long]$CandidateSeedBase,
        [long]$EvaluationSeedBase,
        [long]$ChildPolicySeedBase
    )
    $splitShardCount = [int]($Roots / $RootsPerShard)
    for ($splitShard = 0; $splitShard -lt $splitShardCount; $splitShard += 1) {
        $globalShard = $shardSpecs.Count
        $rootOffset = $splitShard * $RootsPerShard
        $seedStart = $SeedBase + ([long]$rootOffset * $SeedStride)
        $candidateSeed = $CandidateSeedBase + ([long]$splitShard * $SeedStride)
        $evaluationSeed = $EvaluationSeedBase + ([long]$splitShard * $SeedStride)
        $childPolicySeed = $ChildPolicySeedBase + ([long]$splitShard * $SeedStride)
        $shardSpecs.Add([ordered]@{
            schema = "hu_m42_spot_shard_v1"
            shard = $globalShard
            split = $Split
            split_shard = $splitShard
            root_offset = $rootOffset
            roots = $RootsPerShard
            seed_start = $seedStart
            seed_stride = $SeedStride
            candidate_seed = $candidateSeed
            evaluation_seed = $evaluationSeed
            child_policy_seed = $childPolicySeed
            candidate_samples = $CandidateSamples
            evaluation_samples = $EvaluationSamples
            output_prefix = ("{0}_shard_{1:D3}_roots{2:D2}_seed{3}" -f $Split, $splitShard, $RootsPerShard, $seedStart)
            root_profiles = $RootProfiles
            root_profile_weights = $RootProfileWeights
            baseline_profile = $BaselineProfile
            t2_profile = $T2Profile
        })
    }
}

Add-SplitShards -Split "train" -Roots $TrainRoots -SeedBase $TrainSeedBase `
    -CandidateSeedBase $TrainCandidateSeedBase -EvaluationSeedBase $TrainEvaluationSeedBase `
    -ChildPolicySeedBase $TrainChildPolicySeedBase
Add-SplitShards -Split "calibration" -Roots $CalibrationRoots -SeedBase $CalibrationSeedBase `
    -CandidateSeedBase $CalibrationCandidateSeedBase -EvaluationSeedBase $CalibrationEvaluationSeedBase `
    -ChildPolicySeedBase $CalibrationChildPolicySeedBase
Add-SplitShards -Split "locked_holdout" -Roots $LockedHoldoutRoots -SeedBase $LockedHoldoutSeedBase `
    -CandidateSeedBase $LockedCandidateSeedBase -EvaluationSeedBase $LockedEvaluationSeedBase `
    -ChildPolicySeedBase $LockedChildPolicySeedBase

function Convert-ShardSelection {
    param([string[]]$Values, [string]$ParameterName)
    $indices = @()
    foreach ($value in $Values) {
        foreach ($part in ($value -split ',')) {
            $trimmed = $part.Trim()
            if ($trimmed) {
                $parsed = 0
                if (-not [int]::TryParse($trimmed, [ref]$parsed)) {
                    throw "$ParameterName contains a non-integer shard: $trimmed"
                }
                $indices += $parsed
            }
        }
    }
    return @($indices | Sort-Object -Unique)
}

$initialShardIndices = @(Convert-ShardSelection -Values $InitialShards -ParameterName "InitialShards")
$relaunchShardIndices = @(Convert-ShardSelection -Values $StartShards -ParameterName "StartShards")
$explicitInitialLaunch = $initialShardIndices.Count -gt 0
$explicitRelaunch = $relaunchShardIndices.Count -gt 0
if ($explicitInitialLaunch -and $explicitRelaunch) {
    throw "InitialShards and StartShards are mutually exclusive"
}
if ($explicitRelaunch) {
    $selectedShardIndices = $relaunchShardIndices
}
elseif ($explicitInitialLaunch) {
    $selectedShardIndices = $initialShardIndices
}
else {
    $selectedShardIndices = @(0..($shardSpecs.Count - 1))
}
if ($DryRun -or -not $explicitRelaunch) {
    foreach ($index in $selectedShardIndices) {
        if ($index -lt 0 -or $index -ge $shardSpecs.Count) {
            $selectionName = $(if ($explicitRelaunch) { "StartShards" } else { "InitialShards" })
            throw "$selectionName values must be in [0, $($shardSpecs.Count)): $index"
        }
    }
}

$dryRunResult = [ordered]@{
    execution = "dry_run"
    schema = "hu_m42_spot_dry_run_v1"
    run_name = $RunName
    project_id = $ProjectId
    bucket = $Bucket
    total_roots = $TrainRoots + $CalibrationRoots + $LockedHoldoutRoots
    split_roots = [ordered]@{
        train = $TrainRoots
        calibration = $CalibrationRoots
        locked_holdout = $LockedHoldoutRoots
    }
    roots_per_shard = $RootsPerShard
    total_shards = $shardSpecs.Count
    initial_shards = $selectedShardIndices
    selected_shards = $selectedShardIndices
    candidate_samples = $CandidateSamples
    evaluation_samples = $EvaluationSamples
    native_batch_threads = $NativeBatchThreads
    root_profiles = $RootProfiles
    root_profile_weights = $RootProfileWeights
    baseline_profile = $BaselineProfile
    t2_profile = $T2Profile
    machine_type = $MachineType
    boot_disk_type = $BootDiskType
    fallback_machine_types = $FallbackMachineTypes
    fallback_boot_disk_type = $FallbackBootDiskType
    zones = $Zones
    spot = $true
    termination_action = "DELETE"
    self_delete = -not $NoSelfDelete
    create_instances = [bool]$CreateInstances
    relaunch_existing_run = $explicitRelaunch
    initial_launch_subset = $explicitInitialLaunch
    allow_native_source_build_fallback = [bool]$AllowNativeSourceBuildFallback
    required_model_count = $RequiredModels.Count
    required_model_bytes = [long](($modelPlan | Measure-Object bytes -Sum).Sum)
    required_models = $modelPlan
    required_native_binary_count = $nativePlan.Count
    required_native_binary_bytes = [long](($nativePlan | Measure-Object bytes -Sum).Sum)
    required_native_binaries = $nativePlan
    package_source = "current_dirty_worktree"
    shards = $shardSpecs
}
if ($DryRun) {
    $dryRunResult | ConvertTo-Json -Depth 10
    exit 0
}

$cloudSdkGcloud = Join-Path $env:LOCALAPPDATA "Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
if (Test-Path $cloudSdkGcloud) {
    Set-Alias -Name gcloud -Value $cloudSdkGcloud -Scope Script
}
elseif (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    throw "gcloud not found in PATH or at $cloudSdkGcloud"
}

$runDir = Join-Path $repoRoot ("outputs/gcp_runs/{0}" -f $RunName)
$packageDir = Join-Path $runDir "package_src"
$packagePath = Join-Path $runDir "ofc_regular_hu_m42_source.zip"
$startupPath = Join-Path $runDir "startup_hu_m42_spot.sh"
$manifestPath = Join-Path $runDir "manifest.json"
$shardManifestPath = Join-Path $runDir "shards_manifest.jsonl"
$modelManifestPath = Join-Path $runDir "source_model_manifest.json"
$nativeManifestPath = Join-Path $runDir "source_native_manifest.json"
$prefix = "gs://$Bucket/runs/$RunName"
$sourceUri = "$prefix/source/ofc_regular_hu_m42_source.zip"
$startupUri = "$prefix/source/startup_hu_m42_spot.sh"
$manifestUri = "$prefix/manifest.json"
$shardManifestUri = "$prefix/source/shards_manifest.jsonl"
$modelManifestUri = "$prefix/source/source_model_manifest.json"
$nativeManifestUri = "$prefix/source/source_native_manifest.json"

New-Item -ItemType Directory -Force -Path $runDir | Out-Null
$savedErrorActionPreference = $ErrorActionPreference
try {
    # Windows PowerShell 5 wraps native stderr as an ErrorRecord. With the
    # script-wide Stop policy, the expected "object absent" probe would abort
    # before LASTEXITCODE can be classified.
    $ErrorActionPreference = "Continue"
    $manifestProbe = @(& gcloud storage ls $manifestUri --project $ProjectId 2>&1)
    $manifestProbeExit = $LASTEXITCODE
}
finally {
    $ErrorActionPreference = $savedErrorActionPreference
}
$manifestProbeText = ($manifestProbe | Out-String)
if ($manifestProbeExit -eq 0) {
    $remoteManifestExists = @($manifestUri)
}
elseif ($manifestProbeText -match '(?i)(not found|no urls matched|matched no objects|404)') {
    $remoteManifestExists = @()
}
else {
    throw "Unable to determine whether the frozen run exists; refusing fail-open upload: $manifestProbeText"
}
$strictResume = $explicitRelaunch -or ($remoteManifestExists -contains $manifestUri)
if ($explicitRelaunch -and $remoteManifestExists -notcontains $manifestUri) {
    throw "StartShards is relaunch-only, but the frozen remote run does not exist: $manifestUri"
}
if ($explicitInitialLaunch -and $remoteManifestExists -contains $manifestUri) {
    throw "InitialShards is new-run-only, but the frozen remote run already exists: $manifestUri"
}

if ($strictResume) {
    # A run is immutable after its first upload. Relaunch downloads and verifies
    # the frozen source, startup script, and manifests, then creates only the
    # requested missing workers. No GCS object is overwritten in this branch.
    gcloud storage cp $manifestUri $manifestPath --project $ProjectId | Out-Null
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    if ($manifest.schema -ne "hu_m42_spot_manifest_v1" -or
        [string]$manifest.run_name -ne $RunName -or
        [string]$manifest.project_id -ne $ProjectId -or
        [string]$manifest.bucket -ne $Bucket) {
        throw "Frozen M4.2 run identity/schema mismatch"
    }
    $sourceUri = [string]$manifest.source_uri
    $startupUri = [string]$manifest.startup_uri
    $shardManifestUri = [string]$manifest.shards_manifest_uri
    $modelManifestUri = [string]$manifest.model_manifest_uri
    $nativeManifestUri = [string]$manifest.native_manifest_uri
    foreach ($uri in @($sourceUri, $startupUri, $shardManifestUri, $modelManifestUri, $nativeManifestUri)) {
        if (-not $uri -or -not $uri.StartsWith("$prefix/")) {
            throw "Frozen M4.2 manifest contains an invalid run object URI: $uri"
        }
    }
    gcloud storage cp $sourceUri $packagePath --project $ProjectId | Out-Null
    gcloud storage cp $startupUri $startupPath --project $ProjectId | Out-Null
    gcloud storage cp $shardManifestUri $shardManifestPath --project $ProjectId | Out-Null
    gcloud storage cp $modelManifestUri $modelManifestPath --project $ProjectId | Out-Null
    gcloud storage cp $nativeManifestUri $nativeManifestPath --project $ProjectId | Out-Null
    $manifestSha256 = Get-Sha256 $manifestPath
    $sourceSha256 = Get-Sha256 $packagePath
    $startupSha256 = Get-Sha256 $startupPath
    $shardManifestSha256 = Get-Sha256 $shardManifestPath
    $modelManifestSha256 = Get-Sha256 $modelManifestPath
    $nativeManifestSha256 = Get-Sha256 $nativeManifestPath
    if ($sourceSha256 -ne [string]$manifest.source_sha256 -or
        $startupSha256 -ne [string]$manifest.startup_sha256 -or
        $shardManifestSha256 -ne [string]$manifest.shards_manifest_sha256 -or
        $modelManifestSha256 -ne [string]$manifest.model_manifest_sha256 -or
        $nativeManifestSha256 -ne [string]$manifest.native_manifest_sha256) {
        throw "Frozen M4.2 source/startup/manifest hash verification failed"
    }

    $frozenModelManifest = Get-Content -LiteralPath $modelManifestPath -Raw | ConvertFrom-Json
    if ($frozenModelManifest.schema -ne "hu_m42_source_model_manifest_v1" -or
        [int]$frozenModelManifest.model_count -ne 11 -or
        [int]$manifest.required_model_count -ne 11) {
        throw "Frozen M4.2 run does not have the exact 11-model contract"
    }
    $frozenDeclaredModels = @{}
    foreach ($row in $manifest.required_models) { $frozenDeclaredModels[[string]$row.path] = $row }
    foreach ($row in $frozenModelManifest.models) {
        $path = [string]$row.path
        if (-not $frozenDeclaredModels.ContainsKey($path)) {
            throw "Frozen model is absent from the run manifest: $path"
        }
        $declared = $frozenDeclaredModels[$path]
        if ([long]$row.bytes -ne [long]$declared.bytes -or
            [string]$row.sha256 -ne [string]$declared.sha256) {
            throw "Frozen model manifest mismatch: $path"
        }
    }
    $modelEntries = @($frozenModelManifest.models)

    $frozenNativeManifest = Get-Content -LiteralPath $nativeManifestPath -Raw | ConvertFrom-Json
    $expectedNativePaths = @($RequiredNativeBinaries | Sort-Object)
    $frozenNativePaths = @($frozenNativeManifest.binaries | ForEach-Object { [string]$_.path } | Sort-Object)
    if ($frozenNativeManifest.schema -ne "hu_m42_source_native_manifest_v1" -or
        [int]$frozenNativeManifest.binary_count -ne 2 -or
        [int]$manifest.required_native_binary_count -ne 2 -or
        (($expectedNativePaths -join ',') -ne ($frozenNativePaths -join ','))) {
        throw "Frozen M4.2 run does not have the exact two-native-binary contract"
    }
    $frozenDeclaredNative = @{}
    foreach ($row in $manifest.required_native_binaries) { $frozenDeclaredNative[[string]$row.path] = $row }
    foreach ($row in $frozenNativeManifest.binaries) {
        $path = [string]$row.path
        if (-not $frozenDeclaredNative.ContainsKey($path)) {
            throw "Frozen native binary is absent from the run manifest: $path"
        }
        $declared = $frozenDeclaredNative[$path]
        if ([long]$row.bytes -ne [long]$declared.bytes -or
            [string]$row.sha256 -ne [string]$declared.sha256) {
            throw "Frozen native manifest mismatch: $path"
        }
    }
    $nativeEntries = @($frozenNativeManifest.binaries)

    $frozenSpecs = New-Object System.Collections.Generic.List[object]
    foreach ($line in [System.IO.File]::ReadLines((Resolve-Path -LiteralPath $shardManifestPath))) {
        if ($line.Trim()) { $frozenSpecs.Add(($line | ConvertFrom-Json)) }
    }
    if ($frozenSpecs.Count -ne [int]$manifest.total_shards) {
        throw "Frozen M4.2 shard manifest count mismatch"
    }
    $shardSpecs = $frozenSpecs
    if (-not $explicitRelaunch) {
        if ($CreateInstances) {
            throw "Existing frozen run creation requires explicit StartShards; refusing to launch every shard implicitly"
        }
        $selectedShardIndices = @()
    }
    foreach ($index in $selectedShardIndices) {
        if ($index -lt 0 -or $index -ge $shardSpecs.Count) {
            throw "StartShards values must be in [0, $($shardSpecs.Count)): $index"
        }
    }

    # A deliberately partial initial launch is a canary contract. Do not fan
    # out beyond it until shard 0 has either committed DONE or uploaded a
    # resumable heartbeat+checkpoint proving at least one completed root.
    $frozenInitialShards = @($manifest.initial_shards | ForEach-Object { [int]$_ })
    $promotionRequested = $false
    if ($explicitRelaunch -and $frozenInitialShards.Count -lt $shardSpecs.Count) {
        foreach ($index in $selectedShardIndices) {
            if ($frozenInitialShards -notcontains $index) { $promotionRequested = $true }
        }
    }
    if ($promotionRequested) {
        if ($frozenInitialShards -notcontains 0) {
            throw "Canary promotion requires shard 0 in the frozen initial_shards contract"
        }
        $canarySpec = $shardSpecs[0]
        $canaryOutputPrefix = [string]$canarySpec.output_prefix
        $canaryDoneUri = "$prefix/results/$canaryOutputPrefix/DONE"
        $savedErrorActionPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = "Continue"
            $canaryDoneProbe = @(& gcloud storage ls $canaryDoneUri --project $ProjectId 2>&1)
            $canaryDoneProbeExit = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $savedErrorActionPreference
        }
        $canaryDoneProbeText = ($canaryDoneProbe | Out-String)
        $canaryDoneExists = $canaryDoneProbeExit -eq 0
        if (-not $canaryDoneExists -and
            $canaryDoneProbeText -notmatch '(?i)(not found|no urls matched|matched no objects|404)') {
            throw "Unable to verify shard 0 DONE state; refusing fan-out: $canaryDoneProbeText"
        }
        if (-not $canaryDoneExists) {
            $canaryHeartbeatPath = Join-Path $runDir "canary_heartbeat.json"
            $canaryCheckpointPath = Join-Path $runDir "canary_checkpoint.json"
            $canaryResumeConsistent = $false
            for ($attempt = 1; $attempt -le 6 -and -not $canaryResumeConsistent; $attempt += 1) {
                Remove-Item -LiteralPath $canaryHeartbeatPath, $canaryCheckpointPath -Force -ErrorAction SilentlyContinue
                $savedErrorActionPreference = $ErrorActionPreference
                try {
                    $ErrorActionPreference = "Continue"
                    & gcloud storage cp "$prefix/resume/$canaryOutputPrefix/heartbeat.json" $canaryHeartbeatPath --project $ProjectId *> $null
                    $heartbeatCopyExit = $LASTEXITCODE
                    & gcloud storage cp "$prefix/resume/$canaryOutputPrefix/checkpoint.json" $canaryCheckpointPath --project $ProjectId *> $null
                    $checkpointCopyExit = $LASTEXITCODE
                }
                finally {
                    $ErrorActionPreference = $savedErrorActionPreference
                }
                if ($heartbeatCopyExit -eq 0 -and $checkpointCopyExit -eq 0 -and
                    (Test-Path -LiteralPath $canaryHeartbeatPath) -and
                    (Test-Path -LiteralPath $canaryCheckpointPath)) {
                    try {
                        $canaryHeartbeat = Get-Content -LiteralPath $canaryHeartbeatPath -Raw | ConvertFrom-Json
                        $canaryCheckpoint = Get-Content -LiteralPath $canaryCheckpointPath -Raw | ConvertFrom-Json
                        $canaryResumeConsistent = (
                            $canaryHeartbeat.schema -eq "hu_m4_t1_second_heartbeat_v1" -and
                            $canaryCheckpoint.schema -eq "hu_m4_t1_second_checkpoint_v1" -and
                            [int]$canaryHeartbeat.completed_roots -ge 1 -and
                            [int]$canaryHeartbeat.completed_roots -eq [int]$canaryCheckpoint.completed_roots -and
                            [string]$canaryHeartbeat.partial_sha256 -eq [string]$canaryCheckpoint.partial_sha256
                        )
                    }
                    catch { $canaryResumeConsistent = $false }
                }
                if (-not $canaryResumeConsistent -and $attempt -lt 6) { Start-Sleep -Seconds 5 }
            }
            if (-not $canaryResumeConsistent) {
                throw "Shard 0 canary has not produced a consistent resumable root; refusing fan-out"
            }
        }
    }

    function Assert-BoundEqual {
        param([string]$ParameterName, $Actual, $Frozen)
        if ($scriptBoundParameters.ContainsKey($ParameterName) -and [string]$Actual -ne [string]$Frozen) {
            throw "Relaunch parameter $ParameterName conflicts with the frozen run manifest"
        }
    }
    Assert-BoundEqual "TrainRoots" $TrainRoots $manifest.split_roots.train
    Assert-BoundEqual "CalibrationRoots" $CalibrationRoots $manifest.split_roots.calibration
    Assert-BoundEqual "LockedHoldoutRoots" $LockedHoldoutRoots $manifest.split_roots.locked_holdout
    Assert-BoundEqual "RootsPerShard" $RootsPerShard $manifest.roots_per_shard
    Assert-BoundEqual "SeedStride" $SeedStride $manifest.seed_stride
    Assert-BoundEqual "TrainSeedBase" $TrainSeedBase $manifest.seed_bases.train
    Assert-BoundEqual "CalibrationSeedBase" $CalibrationSeedBase $manifest.seed_bases.calibration
    Assert-BoundEqual "LockedHoldoutSeedBase" $LockedHoldoutSeedBase $manifest.seed_bases.locked_holdout
    Assert-BoundEqual "TrainCandidateSeedBase" $TrainCandidateSeedBase $manifest.candidate_seed_bases.train
    Assert-BoundEqual "CalibrationCandidateSeedBase" $CalibrationCandidateSeedBase $manifest.candidate_seed_bases.calibration
    Assert-BoundEqual "LockedCandidateSeedBase" $LockedCandidateSeedBase $manifest.candidate_seed_bases.locked_holdout
    Assert-BoundEqual "TrainEvaluationSeedBase" $TrainEvaluationSeedBase $manifest.evaluation_seed_bases.train
    Assert-BoundEqual "CalibrationEvaluationSeedBase" $CalibrationEvaluationSeedBase $manifest.evaluation_seed_bases.calibration
    Assert-BoundEqual "LockedEvaluationSeedBase" $LockedEvaluationSeedBase $manifest.evaluation_seed_bases.locked_holdout
    Assert-BoundEqual "TrainChildPolicySeedBase" $TrainChildPolicySeedBase $manifest.child_policy_seed_bases.train
    Assert-BoundEqual "CalibrationChildPolicySeedBase" $CalibrationChildPolicySeedBase $manifest.child_policy_seed_bases.calibration
    Assert-BoundEqual "LockedChildPolicySeedBase" $LockedChildPolicySeedBase $manifest.child_policy_seed_bases.locked_holdout
    Assert-BoundEqual "CandidateSamples" $CandidateSamples $manifest.candidate_samples
    Assert-BoundEqual "EvaluationSamples" $EvaluationSamples $manifest.evaluation_samples
    Assert-BoundEqual "NativeBatchThreads" $NativeBatchThreads $manifest.native_batch_threads
    Assert-BoundEqual "SyncIntervalSeconds" $SyncIntervalSeconds $manifest.sync_interval_seconds
    Assert-BoundEqual "MachineType" $MachineType $manifest.machine_type
    Assert-BoundEqual "BootDiskType" $BootDiskType $manifest.boot_disk_type
    Assert-BoundEqual "FallbackBootDiskType" $FallbackBootDiskType $manifest.fallback_boot_disk_type
    Assert-BoundEqual "BootDiskGb" $BootDiskGb $manifest.boot_disk_gb
    if ($scriptBoundParameters.ContainsKey("Zones") -and
        (($Zones -join ',') -ne (@($manifest.zones) -join ','))) {
        throw "Relaunch parameter Zones conflicts with the frozen run manifest"
    }
    if ($scriptBoundParameters.ContainsKey("FallbackMachineTypes") -and
        (($FallbackMachineTypes -join ',') -ne (@($manifest.fallback_machine_types) -join ','))) {
        throw "Relaunch parameter FallbackMachineTypes conflicts with the frozen run manifest"
    }
    if ($scriptBoundParameters.ContainsKey("NoSelfDelete") -and
        ((-not [bool]$NoSelfDelete) -ne [bool]$manifest.self_delete)) {
        throw "Relaunch NoSelfDelete conflicts with the frozen run manifest"
    }
    if ($scriptBoundParameters.ContainsKey("AllowNativeSourceBuildFallback") -and
        ([bool]$AllowNativeSourceBuildFallback -ne [bool]$manifest.allow_native_source_build_fallback)) {
        throw "Relaunch AllowNativeSourceBuildFallback conflicts with the frozen run manifest"
    }

    # Launch settings always come from the verified frozen manifest, including
    # custom settings not repeated on a missing-shard relaunch command.
    $MachineType = [string]$manifest.machine_type
    $BootDiskType = [string]$manifest.boot_disk_type
    $FallbackMachineTypes = @($manifest.fallback_machine_types)
    $FallbackBootDiskType = [string]$manifest.fallback_boot_disk_type
    $Zones = @($manifest.zones)
    $BootDiskGb = [int]$manifest.boot_disk_gb
    $SyncIntervalSeconds = [int]$manifest.sync_interval_seconds
    $NativeBatchThreads = [int]$manifest.native_batch_threads
    $NoSelfDelete = -not [bool]$manifest.self_delete
    $AllowNativeSourceBuildFallback = [bool]$manifest.allow_native_source_build_fallback
}
else {
if (Test-Path $packageDir) { Remove-Item -LiteralPath $packageDir -Recurse -Force }
New-Item -ItemType Directory -Force -Path $packageDir | Out-Null

foreach ($item in @("pyproject.toml", "Cargo.toml", "Cargo.lock", "src", "configs")) {
    $source = Join-Path $repoRoot $item
    if (-not (Test-Path -LiteralPath $source)) { throw "Missing source package input: $item" }
    Copy-Item -LiteralPath $source -Destination $packageDir -Recurse
}
$rustDestination = Join-Path $packageDir "rust"
New-Item -ItemType Directory -Force -Path $rustDestination | Out-Null
Copy-Item -LiteralPath (Join-Path $repoRoot "rust/hu_m3_engine") -Destination $rustDestination -Recurse
Copy-Item -LiteralPath (Join-Path $repoRoot "rust/ofc_stage3_feature_encoder") -Destination $rustDestination -Recurse

$modelEntries = @()
foreach ($model in $RequiredModels) {
    $source = Join-Path $repoRoot $model
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Required M4.2 model is missing: $model"
    }
    $destination = Join-Path $packageDir $model
    New-Item -ItemType Directory -Force -Path (Split-Path $destination -Parent) | Out-Null
    Copy-Item -LiteralPath $source -Destination $destination
    $modelEntries += [pscustomobject][ordered]@{
        path = $model
        bytes = [long](Get-Item -LiteralPath $source).Length
        sha256 = Get-Sha256 $source
    }
}
$modelManifest = [ordered]@{
    schema = "hu_m42_source_model_manifest_v1"
    model_count = $modelEntries.Count
    total_bytes = [long](($modelEntries | Measure-Object bytes -Sum).Sum)
    models = $modelEntries
}
Write-Utf8NoBom -Path $modelManifestPath -Text (($modelManifest | ConvertTo-Json -Depth 6) + "`n")

$nativeEntries = @()
foreach ($native in $RequiredNativeBinaries) {
    $source = Join-Path $repoRoot $native
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Required Docker-built Linux native binary is missing: $native"
    }
    Assert-LinuxX8664Elf -Path $source
    Assert-PinnedNativeArtifact -RelativePath $native -Path $source
    $destination = Join-Path $packageDir $native
    New-Item -ItemType Directory -Force -Path (Split-Path $destination -Parent) | Out-Null
    Copy-Item -LiteralPath $source -Destination $destination
    $nativeEntries += [pscustomobject][ordered]@{
        path = $native
        bytes = [long](Get-Item -LiteralPath $source).Length
        sha256 = Get-Sha256 $source
        platform = "linux-x86_64"
        abi = "elf64-et_dyn-x86_64-glibc>=2.34"
        provenance = "local-docker-release-build"
    }
}
$nativeManifest = [ordered]@{
    schema = "hu_m42_source_native_manifest_v1"
    builder_image = "rust@sha256:7d0723df719e7f213b69dc7c8c595985c3f4b060cfbee4f7bc0e347a86fe3b6a"
    build_target = "generic-linux-x86_64"
    target_cpu_native = $false
    build_commands = @(
        "cargo build --release --lib",
        "cargo build -p ofc_hu_m3_engine --release"
    )
    binary_count = $nativeEntries.Count
    total_bytes = [long](($nativeEntries | Measure-Object bytes -Sum).Sum)
    binaries = $nativeEntries
}
Write-Utf8NoBom -Path $nativeManifestPath -Text (($nativeManifest | ConvertTo-Json -Depth 6) + "`n")

$shardLines = @($shardSpecs | ForEach-Object { $_ | ConvertTo-Json -Compress -Depth 8 })
Write-Utf8NoBom -Path $shardManifestPath -Text (($shardLines -join "`n") + "`n")
Copy-Item -LiteralPath $shardManifestPath -Destination (Join-Path $packageDir "shards_manifest.jsonl")
Copy-Item -LiteralPath $modelManifestPath -Destination (Join-Path $packageDir "source_model_manifest.json")
Copy-Item -LiteralPath $nativeManifestPath -Destination (Join-Path $packageDir "source_native_manifest.json")
New-ZipWithForwardSlashes -SourceDir $packageDir -DestinationPath $packagePath

$sourceSha256 = Get-Sha256 $packagePath
$shardManifestSha256 = Get-Sha256 $shardManifestPath
$modelManifestSha256 = Get-Sha256 $modelManifestPath
$nativeManifestSha256 = Get-Sha256 $nativeManifestPath

$startup = @'
#!/usr/bin/env bash
set -euo pipefail
export HOME="${HOME:-/root}"

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
meta() { curl -fsS -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }
instance_meta() { curl -fsS -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/$1"; }
json_field() { python3 -c 'import json,sys; print(json.loads(sys.argv[1])[sys.argv[2]])' "$1" "$2"; }

RUN_NAME="$(meta RUN_NAME)"
BUCKET="$(meta BUCKET)"
SHARD_INDEX="$(meta SHARD_INDEX)"
SOURCE_URI="$(meta SOURCE_URI)"
SOURCE_SHA256="$(meta SOURCE_SHA256)"
STARTUP_SHA256="$(meta STARTUP_SHA256)"
MANIFEST_SHA256="$(meta MANIFEST_SHA256)"
SHARD_MANIFEST_SHA256="$(meta SHARD_MANIFEST_SHA256)"
MODEL_MANIFEST_SHA256="$(meta MODEL_MANIFEST_SHA256)"
NATIVE_MANIFEST_SHA256="$(meta NATIVE_MANIFEST_SHA256)"
SYNC_INTERVAL_SECONDS="$(meta SYNC_INTERVAL_SECONDS)"
NATIVE_BATCH_THREADS="$(meta NATIVE_BATCH_THREADS)"
ALLOW_NATIVE_SOURCE_BUILD_FALLBACK="$(meta ALLOW_NATIVE_SOURCE_BUILD_FALLBACK)"
SELF_DELETE="$(meta SELF_DELETE)"
INSTANCE_NAME="$(instance_meta name)"
ZONE_PATH="$(instance_meta zone)"
ZONE="${ZONE_PATH##*/}"
STARTUP_LOG="/tmp/ofc-hu-m42-startup.log"
: > "$STARTUP_LOG"
exec > >(tee -a "$STARTUP_LOG") 2>&1
PREFIX="gs://${BUCKET}/runs/${RUN_NAME}"
STATUS_URI="${PREFIX}/status/shard_${SHARD_INDEX}.json"
WORK_DIR="/opt/ofc-regular-hu-m42"
RESULT_DIR="/tmp/ofc-hu-m42-result"
SYNC_DIR="/tmp/ofc-hu-m42-sync"
STATUS_PATH="/tmp/ofc-hu-m42-status.json"
GENERATOR_PID=""
OUTPUT_PREFIX=""
SPLIT=""

completed_roots() {
  if [[ ! -f "${RESULT_DIR}/checkpoint.json" ]]; then echo 0; return; fi
  python3 - "${RESULT_DIR}/checkpoint.json" <<'PY'
import json,sys
try:
    value=json.load(open(sys.argv[1], encoding="utf-8")).get("completed_roots", 0)
    print(int(value))
except Exception:
    print(0)
PY
}

write_status() {
  local state="$1"
  local exit_code="${2:-0}"
  local completed="$(completed_roots)"
  python3 - "$STATUS_PATH" "$RUN_NAME" "$SHARD_INDEX" "$SPLIT" "$state" "$exit_code" "$completed" "$INSTANCE_NAME" "$ZONE" "$OUTPUT_PREFIX" <<'PY'
import json,sys,time
path,run,shard,split,state,exit_code,completed,instance,zone,prefix=sys.argv[1:]
payload={
    "schema":"hu_m42_spot_status_v1", "run_name":run, "shard":int(shard),
    "split":split, "status":state, "exit_code":int(exit_code),
    "completed_roots":int(completed), "instance":instance, "zone":zone,
    "output_prefix":prefix, "updated_unix_seconds":time.time(),
}
open(path,"w",encoding="utf-8").write(json.dumps(payload,sort_keys=True)+"\n")
PY
  gcloud storage cp "$STATUS_PATH" "$STATUS_URI" >/dev/null || true
}

sync_resume() {
  [[ -n "$OUTPUT_PREFIX" ]] || return 0
  local partial="${RESULT_DIR}/teacher.jsonl.partial"
  local checkpoint="${RESULT_DIR}/checkpoint.json"
  local heartbeat="${RESULT_DIR}/heartbeat.json"
  [[ -s "$partial" && -s "$checkpoint" ]] || return 0
  rm -rf "$SYNC_DIR"
  mkdir -p "$SYNC_DIR"
  cp "$checkpoint" "${SYNC_DIR}/checkpoint.json"
  cp "$partial" "${SYNC_DIR}/teacher.jsonl.partial"
  local expected actual
  expected="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8")).get("partial_sha256",""))' "${SYNC_DIR}/checkpoint.json")"
  actual="$(sha256sum "${SYNC_DIR}/teacher.jsonl.partial" | awk '{print $1}')"
  if [[ -z "$expected" || "$expected" != "$actual" ]]; then
    log "skip inconsistent resume snapshot shard=$SHARD_INDEX expected=$expected actual=$actual"
    return 0
  fi
  if [[ -s "$heartbeat" ]]; then cp "$heartbeat" "${SYNC_DIR}/heartbeat.json"; fi
  local resume_uri="${PREFIX}/resume/${OUTPUT_PREFIX}"
  gcloud storage cp "${SYNC_DIR}/teacher.jsonl.partial" "${resume_uri}/teacher.jsonl.partial" >/dev/null
  gcloud storage cp "${SYNC_DIR}/checkpoint.json" "${resume_uri}/checkpoint.json" >/dev/null
  if [[ -s "${SYNC_DIR}/heartbeat.json" ]]; then
    gcloud storage cp "${SYNC_DIR}/heartbeat.json" "${resume_uri}/heartbeat.json" >/dev/null || true
  fi
  write_status running 0
}

upload_startup_log() {
  [[ -s "$STARTUP_LOG" ]] || return 0
  local startup_log_uri="${PREFIX}/logs/startup_shard_${SHARD_INDEX}.log"
  if command -v gcloud >/dev/null 2>&1; then
    gcloud storage cp "$STARTUP_LOG" "$startup_log_uri" >/dev/null 2>&1 && return 0
  fi
  # Preserve the earliest apt/pip/native failure even when gcloud installation
  # itself failed. The VM service account token can upload directly to GCS.
  local token object encoded
  token="$(curl -fsS -H 'Metadata-Flavor: Google' \
    'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token' | \
    python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')" || return 0
  object="runs/${RUN_NAME}/logs/startup_shard_${SHARD_INDEX}.log"
  encoded="$(python3 -c 'import sys,urllib.parse; print(urllib.parse.quote(sys.argv[1],safe=""))' "$object")" || return 0
  curl -fsS -X POST -H "Authorization: Bearer ${token}" -H 'Content-Type: text/plain' \
    --data-binary @"$STARTUP_LOG" \
    "https://storage.googleapis.com/upload/storage/v1/b/${BUCKET}/o?uploadType=media&name=${encoded}" \
    >/dev/null 2>&1 || true
}

cleanup() {
  local code=$?
  set +e
  if [[ "$code" -ne 0 ]]; then
    sync_resume
    [[ -f "${RESULT_DIR}/run.log" ]] && gcloud storage cp "${RESULT_DIR}/run.log" "${PREFIX}/logs/shard_${SHARD_INDEX}.log" >/dev/null
    write_status failed "$code"
    upload_startup_log
  fi
  if [[ "$SELF_DELETE" == "1" ]] && command -v gcloud >/dev/null 2>&1; then
    gcloud compute instances delete "$INSTANCE_NAME" --zone "$ZONE" --quiet >/dev/null 2>&1 || sudo shutdown -h now
  elif [[ "$code" -ne 0 ]]; then
    sudo shutdown -h now || true
  fi
  exit "$code"
}
trap cleanup EXIT

export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -y
sudo apt-get install -y unzip curl ca-certificates python3 python3-venv python3-pip
if ! command -v gcloud >/dev/null 2>&1; then
  sudo apt-get install -y apt-transport-https gnupg
  curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
  echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list >/dev/null
  sudo apt-get update -y
  sudo apt-get install -y google-cloud-cli
fi
sudo rm -rf "$WORK_DIR"
sudo mkdir -p "$WORK_DIR"
sudo chown "$(id -u):$(id -g)" "$WORK_DIR"
gcloud storage cp "$SOURCE_URI" /tmp/ofc_regular_hu_m42_source.zip >/dev/null
ACTUAL_SOURCE_SHA256="$(sha256sum /tmp/ofc_regular_hu_m42_source.zip | awk '{print $1}')"
[[ "$ACTUAL_SOURCE_SHA256" == "$SOURCE_SHA256" ]] || { log "source SHA256 mismatch"; exit 20; }
unzip -q /tmp/ofc_regular_hu_m42_source.zip -d "$WORK_DIR"
cd "$WORK_DIR"
[[ "$(sha256sum shards_manifest.jsonl | awk '{print $1}')" == "$SHARD_MANIFEST_SHA256" ]] || { log "embedded shard manifest SHA256 mismatch"; exit 27; }
[[ "$(sha256sum source_model_manifest.json | awk '{print $1}')" == "$MODEL_MANIFEST_SHA256" ]] || { log "embedded model manifest SHA256 mismatch"; exit 28; }
[[ "$(sha256sum source_native_manifest.json | awk '{print $1}')" == "$NATIVE_MANIFEST_SHA256" ]] || { log "embedded native manifest SHA256 mismatch"; exit 29; }

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "numpy==2.2.6" "scikit-learn==1.8.0"
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.6.0"
export PYTHONPATH="$WORK_DIR/src"
export OMP_NUM_THREADS="$NATIVE_BATCH_THREADS"
export MKL_NUM_THREADS="$NATIVE_BATCH_THREADS"
export OPENBLAS_NUM_THREADS="$NATIVE_BATCH_THREADS"
export OFC_HU_M3_BATCH_THREADS="$NATIVE_BATCH_THREADS"

python - <<'PY'
import hashlib,json,pathlib
manifest=json.loads(pathlib.Path("source_model_manifest.json").read_text(encoding="utf-8"))
assert manifest["schema"] == "hu_m42_source_model_manifest_v1"
assert manifest["model_count"] == 11
for row in manifest["models"]:
    path=pathlib.Path(row["path"])
    assert path.is_file(), path
    assert path.stat().st_size == row["bytes"], path
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], path
print("m42_model_manifest=verified")
PY

python - <<'PY'
import hashlib,json,pathlib
manifest=json.loads(pathlib.Path("source_native_manifest.json").read_text(encoding="utf-8"))
assert manifest["schema"] == "hu_m42_source_native_manifest_v1"
assert manifest["builder_image"] == "rust@sha256:7d0723df719e7f213b69dc7c8c595985c3f4b060cfbee4f7bc0e347a86fe3b6a"
assert manifest["build_target"] == "generic-linux-x86_64"
assert manifest["target_cpu_native"] is False
assert manifest["binary_count"] == 2
expected={
    "target/release/libofc_stage3_feature_encoder.so",
    "target/release/libofc_hu_m3_engine.so",
}
assert {row["path"] for row in manifest["binaries"]} == expected
for row in manifest["binaries"]:
    path=pathlib.Path(row["path"])
    assert row["platform"] == "linux-x86_64"
    assert row["abi"] == "elf64-et_dyn-x86_64-glibc>=2.34"
    assert path.is_file(), path
    assert path.stat().st_size == row["bytes"], path
    payload=path.read_bytes()
    assert payload[:6] == b"\x7fELF\x02\x01", path
    assert payload[16:20] == b"\x03\x00\x3e\x00", path
    assert hashlib.sha256(payload).hexdigest() == row["sha256"], path
print("m42_native_manifest=verified")
PY

native_smoke() {
python - <<'PY'
from ofc_regular.hu_m3_rust import engine_version, load_native_engine
from ofc_regular.hu_turn3_stage3_feature_rust import rust_direct_available
print("hu_m3_engine=" + engine_version(library=load_native_engine()))
assert rust_direct_available(), "Pinned Stage3 feature encoder did not load"
print("stage3_feature_encoder=available")
PY
}

if ! native_smoke; then
  if [[ "$ALLOW_NATIVE_SOURCE_BUILD_FALLBACK" != "1" ]]; then
    log "pinned native library smoke failed and source-build fallback is disabled"
    exit 30
  fi
  log "explicit native source-build fallback requested"
  sudo apt-get install -y build-essential
  if [[ ! -x "$HOME/.cargo/bin/rustup" ]]; then
    curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
  fi
  source "$HOME/.cargo/env"
  cargo build --release --lib
  cargo build -p ofc_hu_m3_engine --release
  native_smoke
fi

SPEC="$(sed -n "$((SHARD_INDEX + 1))p" shards_manifest.jsonl)"
[[ -n "$SPEC" ]] || { log "missing shard spec index=$SHARD_INDEX"; exit 21; }
SPEC_SHARD="$(json_field "$SPEC" shard)"
[[ "$SPEC_SHARD" == "$SHARD_INDEX" ]] || { log "shard manifest index mismatch"; exit 22; }
SPLIT="$(json_field "$SPEC" split)"
ROOTS="$(json_field "$SPEC" roots)"
SEED_START="$(json_field "$SPEC" seed_start)"
SEED_STRIDE="$(json_field "$SPEC" seed_stride)"
CANDIDATE_SEED="$(json_field "$SPEC" candidate_seed)"
EVALUATION_SEED="$(json_field "$SPEC" evaluation_seed)"
CHILD_POLICY_SEED="$(json_field "$SPEC" child_policy_seed)"
CANDIDATE_SAMPLES="$(json_field "$SPEC" candidate_samples)"
EVALUATION_SAMPLES="$(json_field "$SPEC" evaluation_samples)"
OUTPUT_PREFIX="$(json_field "$SPEC" output_prefix)"
DONE_URI="${PREFIX}/results/${OUTPUT_PREFIX}/DONE"
RESULT_URI="${PREFIX}/results/${OUTPUT_PREFIX}"
RESUME_URI="${PREFIX}/resume/${OUTPUT_PREFIX}"

if gcloud storage ls "$DONE_URI" >/dev/null 2>&1; then
  log "DONE already exists; shard=$SHARD_INDEX is idempotently complete"
  write_status complete 0
  exit 0
fi

rm -rf "$RESULT_DIR" "$SYNC_DIR"
mkdir -p "$RESULT_DIR" "$SYNC_DIR"
if gcloud storage ls "${RESUME_URI}/checkpoint.json" >/dev/null 2>&1 &&
   gcloud storage ls "${RESUME_URI}/teacher.jsonl.partial" >/dev/null 2>&1; then
  log "download resumable boundary shard=$SHARD_INDEX"
  gcloud storage cp "${RESUME_URI}/checkpoint.json" "${RESULT_DIR}/checkpoint.json" >/dev/null
  gcloud storage cp "${RESUME_URI}/teacher.jsonl.partial" "${RESULT_DIR}/teacher.jsonl.partial" >/dev/null
  gcloud storage cp "${RESUME_URI}/heartbeat.json" "${RESULT_DIR}/heartbeat.json" >/dev/null 2>&1 || true
fi
write_status running 0

IFS=' ' read -r -a ROOT_PROFILE_VALUES <<< "stage19_p0 stage9f_p2 stage7_m5_r10 stage3_baseline random_exact_final"
IFS=' ' read -r -a ROOT_WEIGHT_VALUES <<< "1 1 1 1 1"
args=(
  -B -m ofc_regular.generate_hu_m4_t1_data
  --output "${RESULT_DIR}/teacher.jsonl"
  --checkpoint "${RESULT_DIR}/checkpoint.json"
  --heartbeat "${RESULT_DIR}/heartbeat.json"
  --roots "$ROOTS"
  --seed-start "$SEED_START"
  --seed-stride "$SEED_STRIDE"
  --candidate-samples "$CANDIDATE_SAMPLES"
  --evaluation-samples "$EVALUATION_SAMPLES"
  --candidate-seed "$CANDIDATE_SEED"
  --evaluation-seed "$EVALUATION_SEED"
  --child-policy-seed "$CHILD_POLICY_SEED"
  --root-profiles "${ROOT_PROFILE_VALUES[@]}"
  --root-profile-weights "${ROOT_WEIGHT_VALUES[@]}"
  --baseline-profile stage18_p1
  --t2-profile stage9f_p2
  --batch-child-selectors
  --native-batch-threads "$NATIVE_BATCH_THREADS"
  --opening-lookahead-samples 0
  --split "$SPLIT"
  --run-id "${RUN_NAME}:shard=${SHARD_INDEX}"
)

set +e
python "${args[@]}" > "${RESULT_DIR}/generator_summary.json" 2> "${RESULT_DIR}/run.log" &
GENERATOR_PID=$!
set -e
while kill -0 "$GENERATOR_PID" >/dev/null 2>&1; do
  for (( waited=0; waited<SYNC_INTERVAL_SECONDS; waited+=1 )); do
    kill -0 "$GENERATOR_PID" >/dev/null 2>&1 || break
    sleep 1
  done
  kill -0 "$GENERATOR_PID" >/dev/null 2>&1 && sync_resume || true
done
set +e
wait "$GENERATOR_PID"
EXIT_CODE=$?
set -e
GENERATOR_PID=""
if [[ "$EXIT_CODE" -ne 0 ]]; then
  sync_resume || true
  gcloud storage cp "${RESULT_DIR}/run.log" "${PREFIX}/logs/shard_${SHARD_INDEX}.log" >/dev/null || true
  write_status failed "$EXIT_CODE"
  exit "$EXIT_CODE"
fi

[[ -s "${RESULT_DIR}/teacher.jsonl" ]] || { log "generator completed without teacher output"; exit 23; }
[[ -s "${RESULT_DIR}/checkpoint.json" ]] || { log "generator completed without checkpoint"; exit 24; }
[[ -s "${RESULT_DIR}/heartbeat.json" ]] || { log "generator completed without heartbeat"; exit 25; }
OUTPUT_SHA256="$(sha256sum "${RESULT_DIR}/teacher.jsonl" | awk '{print $1}')"
CHECKPOINT_SHA256="$(sha256sum "${RESULT_DIR}/checkpoint.json" | awk '{print $1}')"
HEARTBEAT_SHA256="$(sha256sum "${RESULT_DIR}/heartbeat.json" | awk '{print $1}')"
OUTPUT_LINES="$(wc -l < "${RESULT_DIR}/teacher.jsonl" | tr -d ' ')"
[[ "$OUTPUT_LINES" == "$ROOTS" ]] || { log "teacher line count mismatch expected=$ROOTS actual=$OUTPUT_LINES"; exit 26; }

python3 - "${RESULT_DIR}/DONE" "$RUN_NAME" "$SHARD_INDEX" "$SPLIT" "$ROOTS" "$OUTPUT_PREFIX" "$OUTPUT_SHA256" "$CHECKPOINT_SHA256" "$HEARTBEAT_SHA256" "$SOURCE_SHA256" "$STARTUP_SHA256" "$MANIFEST_SHA256" "$SHARD_MANIFEST_SHA256" "$MODEL_MANIFEST_SHA256" "$NATIVE_MANIFEST_SHA256" <<'PY'
import json,sys,time
(path,run,shard,split,roots,prefix,output_sha,checkpoint_sha,heartbeat_sha,
 source_sha,startup_sha,manifest_sha,shards_sha,models_sha,native_sha)=sys.argv[1:]
payload={
    "schema":"hu_m42_spot_done_v1", "status":"complete", "run_name":run,
    "shard":int(shard), "split":split, "roots":int(roots),
    "output_prefix":prefix, "output_file":"teacher.jsonl",
    "output_sha256":output_sha, "checkpoint_sha256":checkpoint_sha,
    "heartbeat_sha256":heartbeat_sha, "source_sha256":source_sha,
    "startup_sha256":startup_sha,
    "manifest_sha256":manifest_sha, "shards_manifest_sha256":shards_sha,
    "model_manifest_sha256":models_sha, "native_manifest_sha256":native_sha,
    "completed_unix_seconds":time.time(),
}
open(path,"w",encoding="utf-8").write(json.dumps(payload,sort_keys=True)+"\n")
PY

for file in teacher.jsonl checkpoint.json heartbeat.json generator_summary.json run.log; do
  gcloud storage cp "${RESULT_DIR}/${file}" "${RESULT_URI}/${file}" >/dev/null
done
# DONE is deliberately uploaded last. Its presence is the idempotent commit.
gcloud storage cp "${RESULT_DIR}/DONE" "$DONE_URI" >/dev/null
write_status complete 0
log "complete shard=$SHARD_INDEX split=$SPLIT roots=$ROOTS sha256=$OUTPUT_SHA256"
'@
Write-Utf8NoBom -Path $startupPath -Text $startup
$startupSha256 = Get-Sha256 $startupPath

$manifest = [ordered]@{
    schema = "hu_m42_spot_manifest_v1"
    run_name = $RunName
    project_id = $ProjectId
    bucket = $Bucket
    milestone = "M4.2"
    purpose = "bounded_oof_candidate_quality_gate_not_runtime_promotion"
    package_source = "current_dirty_worktree"
    total_roots = $TrainRoots + $CalibrationRoots + $LockedHoldoutRoots
    split_roots = [ordered]@{
        train = $TrainRoots
        calibration = $CalibrationRoots
        locked_holdout = $LockedHoldoutRoots
    }
    roots_per_shard = $RootsPerShard
    total_shards = $shardSpecs.Count
    initial_shards = $selectedShardIndices
    candidate_samples = $CandidateSamples
    evaluation_samples = $EvaluationSamples
    seed_stride = $SeedStride
    seed_bases = [ordered]@{
        train = $TrainSeedBase
        calibration = $CalibrationSeedBase
        locked_holdout = $LockedHoldoutSeedBase
    }
    candidate_seed_bases = [ordered]@{
        train = $TrainCandidateSeedBase
        calibration = $CalibrationCandidateSeedBase
        locked_holdout = $LockedCandidateSeedBase
    }
    evaluation_seed_bases = [ordered]@{
        train = $TrainEvaluationSeedBase
        calibration = $CalibrationEvaluationSeedBase
        locked_holdout = $LockedEvaluationSeedBase
    }
    child_policy_seed_bases = [ordered]@{
        train = $TrainChildPolicySeedBase
        calibration = $CalibrationChildPolicySeedBase
        locked_holdout = $LockedChildPolicySeedBase
    }
    root_profiles = $RootProfiles
    root_profile_weights = $RootProfileWeights
    baseline_profile = $BaselineProfile
    t2_profile = $T2Profile
    batch_child_selectors = $true
    native_batch_threads = $NativeBatchThreads
    allow_native_source_build_fallback = [bool]$AllowNativeSourceBuildFallback
    opening_lookahead_samples = 0
    machine_type = $MachineType
    boot_disk_type = $BootDiskType
    fallback_machine_types = $FallbackMachineTypes
    fallback_boot_disk_type = $FallbackBootDiskType
    zones = $Zones
    boot_disk_gb = $BootDiskGb
    spot = $true
    instance_termination_action = "DELETE"
    self_delete = -not $NoSelfDelete
    sync_interval_seconds = $SyncIntervalSeconds
    source_uri = $sourceUri
    source_bytes = [long](Get-Item -LiteralPath $packagePath).Length
    source_sha256 = $sourceSha256
    startup_uri = $startupUri
    startup_sha256 = $startupSha256
    shards_manifest_uri = $shardManifestUri
    shards_manifest_sha256 = $shardManifestSha256
    model_manifest_uri = $modelManifestUri
    model_manifest_sha256 = $modelManifestSha256
    native_manifest_uri = $nativeManifestUri
    native_manifest_sha256 = $nativeManifestSha256
    required_model_count = $modelEntries.Count
    required_model_bytes = [long](($modelEntries | Measure-Object bytes -Sum).Sum)
    required_models = $modelEntries
    required_native_binary_count = $nativeEntries.Count
    required_native_binary_bytes = [long](($nativeEntries | Measure-Object bytes -Sum).Sum)
    required_native_binaries = $nativeEntries
    no_runtime_activation = $true
    created_at = (Get-Date).ToUniversalTime().ToString("o")
}
Write-Utf8NoBom -Path $manifestPath -Text (($manifest | ConvertTo-Json -Depth 10) + "`n")
$manifestSha256 = Get-Sha256 $manifestPath

if ($PackageOnly) {
    [pscustomobject]@{
        schema = "hu_m42_spot_package_only_result_v1"
        run_name = $RunName
        package = $packagePath
        source_sha256 = $sourceSha256
        manifest = $manifestPath
        manifest_sha256 = $manifestSha256
        startup = $startupPath
        startup_sha256 = $startupSha256
        total_shards = $shardSpecs.Count
        selected_shards = $selectedShardIndices
        required_native_binary_count = $nativeEntries.Count
    } | ConvertTo-Json -Depth 6
    exit 0
}

gcloud storage cp $packagePath $sourceUri --project $ProjectId | Out-Null
gcloud storage cp $startupPath $startupUri --project $ProjectId | Out-Null
gcloud storage cp $manifestPath $manifestUri --project $ProjectId | Out-Null
gcloud storage cp $shardManifestPath $shardManifestUri --project $ProjectId | Out-Null
gcloud storage cp $modelManifestPath $modelManifestUri --project $ProjectId | Out-Null
gcloud storage cp $nativeManifestPath $nativeManifestUri --project $ProjectId | Out-Null
}

$vmPrefix = Convert-ToVmPrefix $RunName
$instances = @()
foreach ($shardIndex in $selectedShardIndices) {
    $vmName = ("{0}-{1:D3}" -f $vmPrefix, $shardIndex)
    if ($vmName.Length -gt 63) { $vmName = $vmName.Substring(0, 63).Trim('-') }
    $metadata = @(
        "RUN_NAME=$RunName",
        "BUCKET=$Bucket",
        "SHARD_INDEX=$shardIndex",
        "SOURCE_URI=$sourceUri",
        "SOURCE_SHA256=$sourceSha256",
        "STARTUP_SHA256=$startupSha256",
        "MANIFEST_SHA256=$manifestSha256",
        "SHARD_MANIFEST_SHA256=$shardManifestSha256",
        "MODEL_MANIFEST_SHA256=$modelManifestSha256",
        "NATIVE_MANIFEST_SHA256=$nativeManifestSha256",
        "SYNC_INTERVAL_SECONDS=$SyncIntervalSeconds",
        "NATIVE_BATCH_THREADS=$NativeBatchThreads",
        ("ALLOW_NATIVE_SOURCE_BUILD_FALLBACK=" + ($(if ($AllowNativeSourceBuildFallback) { "1" } else { "0" }))),
        ("SELF_DELETE=" + ($(if ($NoSelfDelete) { "0" } else { "1" })))
    ) -join ','
    $instanceRecord = [ordered]@{
        name = $vmName
        shard = $shardIndex
        requested_machine_type = $MachineType
        requested_boot_disk_type = $BootDiskType
        zone = $null
        machine_type = $null
        boot_disk_type = $null
        created = $false
    }
    if ($CreateInstances) {
        $existing = @(gcloud compute instances list --project $ProjectId `
            --filter ("name='" + $vmName + "'") `
            --format "value(name,zone,status)" 2>$null)
        if ($existing.Count -gt 0) {
            if ($SkipExistingInstances) {
                $instanceRecord.zone = (($existing[0] -split '\s+')[1] -split '/')[-1]
                $instances += [pscustomobject]$instanceRecord
                continue
            }
            throw "Instance named $vmName already exists; use a new RunName or -SkipExistingInstances"
        }

        $attempts = @([pscustomobject]@{ machine = $MachineType; disk = $BootDiskType })
        foreach ($fallbackMachine in $FallbackMachineTypes) {
            if ($fallbackMachine -and $fallbackMachine -ne $MachineType) {
                $attempts += [pscustomobject]@{ machine = $fallbackMachine; disk = $FallbackBootDiskType }
            }
        }
        $created = $false
        for ($attemptIndex = 0; $attemptIndex -lt $attempts.Count -and -not $created; $attemptIndex += 1) {
            $attempt = $attempts[$attemptIndex]
            for ($zoneOffset = 0; $zoneOffset -lt $Zones.Count -and -not $created; $zoneOffset += 1) {
                $zone = $Zones[($shardIndex + $zoneOffset) % $Zones.Count]
                $createArgs = @(
                    "compute", "instances", "create", $vmName,
                    "--project", $ProjectId,
                    "--zone", $zone,
                    "--machine-type", $attempt.machine,
                    "--image-family", "ubuntu-2404-lts-amd64",
                    "--image-project", "ubuntu-os-cloud",
                    "--boot-disk-size", ("{0}GB" -f $BootDiskGb),
                    "--boot-disk-type", $attempt.disk,
                    "--boot-disk-auto-delete",
                    "--provisioning-model", "SPOT",
                    "--instance-termination-action", "DELETE",
                    "--maintenance-policy", "TERMINATE",
                    "--scopes", "https://www.googleapis.com/auth/cloud-platform",
                    "--labels", "purpose=hu-m42-teacher,milestone=m42",
                    "--metadata", $metadata,
                    "--metadata-from-file", ("startup-script={0}" -f $startupPath)
                )
                & gcloud @createArgs | Out-Host
                if ($LASTEXITCODE -eq 0) {
                    $created = $true
                    $instanceRecord.zone = $zone
                    $instanceRecord.machine_type = $attempt.machine
                    $instanceRecord.boot_disk_type = $attempt.disk
                    $instanceRecord.created = $true
                }
                else {
                    Write-Warning "Spot create failed for $vmName type=$($attempt.machine) disk=$($attempt.disk) zone=$zone; trying fallback"
                }
            }
        }
        if (-not $created) { throw "All Spot create attempts failed for shard $shardIndex ($vmName)" }
    }
    $instances += [pscustomobject]$instanceRecord
}

[pscustomobject]@{
    schema = "hu_m42_spot_start_result_v1"
    run_name = $RunName
    project_id = $ProjectId
    bucket = $Bucket
    total_roots = $manifest.total_roots
    total_shards = $shardSpecs.Count
    selected_shards = $selectedShardIndices
    machine_type = $MachineType
    boot_disk_type = $BootDiskType
    fallback_machine_types = $FallbackMachineTypes
    fallback_boot_disk_type = $FallbackBootDiskType
    spot = $true
    termination_action = "DELETE"
    create_instances = [bool]$CreateInstances
    source_uri = $sourceUri
    source_sha256 = $sourceSha256
    manifest_uri = $manifestUri
    manifest_sha256 = $manifestSha256
    required_model_count = $modelEntries.Count
    required_model_bytes = $manifest.required_model_bytes
    required_native_binary_count = $nativeEntries.Count
    required_native_binary_bytes = $manifest.required_native_binary_bytes
    instances = $instances
} | ConvertTo-Json -Depth 8
