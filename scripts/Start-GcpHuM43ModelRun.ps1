param(
    [string]$ProjectId = "ofc-solver-485418",
    [string]$Bucket = "pokerhu-ofc-solver-485418-training",
    [string]$RunName = ("regular-hu-m43-fold-model-" + (Get-Date -Format "yyyyMMdd-HHmmss")),
    [string[]]$Train = @(),
    [string[]]$Calibration = @(),
    [string]$M43DataContract = "",
    [string]$M43Plan = "configs/hu_joint_policy_m43_pilot.json",
    [int[]]$StartShards = @(0, 1, 2, 3, 4, 5, 6, 7),
    [string]$MachineType = "c4-highcpu-4",
    [string[]]$FallbackMachineTypes = @("c4-standard-4", "n2-highcpu-4"),
    [string[]]$Zones = @("asia-northeast1-b", "asia-northeast1-c", "us-central1-a"),
    [int]$BootDiskGb = 30,
    [switch]$CreateInstances,
    [switch]$ResumeExisting,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$ExpectedJobs = 30
$OuterFolds = 5
$InnerFoldsPerOuter = 5
$JobsPerShard = 4
$ExpectedShards = 8
$Iterations = 150
$MaxLeafNodes = 31
$LearningRate = 0.05
$L2Regularization = 1.0
$Seed = 2026071801
$PairedSeFloor = 0.50
$PairedHuberAlpha = 0.90
$DownsideQuantile = 0.90
$PositiveGainScoreWeight = 0.25
$DownsideRiskScoreWeight = 0.50
$EnsembleDisagreementScoreWeight = 0.25
$ActionScoreMode = "baseline_paired_delta_risk_ensemble_v3"
$ModelId = "hu-m43-t1-second-regular-hu-m43-c2e16-pilot200-20260713-1810"
$NearBestMargin = 0.5
$MinimumSafeTeacherGain = 0.0
$SafetyCalibratorC = 0.25
$SafetyFitRatio = 0.5
$SafetySplitSeed = 2026071802
$MinimumSafetyFitSamples = 30
$MinimumThresholdLockSamples = 30
$Thresholds = "0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.975,0.99,1"
$MinimumCalibrationFires = 10
$MaximumFalsePositiveRate = 0.30
$MaximumP95Loss = 25.0
$MaximumP99Loss = 40.0
$MaximumMaxLoss = 50.0
$ExpectedPredeclaredReceiptSha256 = "9313bdc6ffa33335032ee3d98e279d123d2f3aa70571114e75ee5f38e2594bea"

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-TextSha256 {
    param([Parameter(Mandatory = $true)][string]$Text)
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [System.Text.UTF8Encoding]::new($false).GetBytes($Text)
        return ([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace("-", "").ToLowerInvariant()
    }
    finally { $sha.Dispose() }
}

function Write-BytesCreateNew {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][byte[]]$Bytes
    )
    $stream = [IO.File]::Open($Path, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
    try { $stream.Write($Bytes, 0, $Bytes.Length) }
    finally { $stream.Dispose() }
}

function Write-Utf8CreateNew {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Text
    )
    Write-BytesCreateNew -Path $Path -Bytes ([Text.UTF8Encoding]::new($false).GetBytes($Text))
}

function Resolve-InputPath {
    param([Parameter(Mandatory = $true)][string]$Path, [Parameter(Mandatory = $true)][string]$RepoRoot)
    $candidate = if ([IO.Path]::IsPathRooted($Path)) { $Path } else { Join-Path $RepoRoot $Path }
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) { throw "Required file is missing: $candidate" }
    return (Resolve-Path -LiteralPath $candidate).Path
}

function Invoke-Gcloud {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { $output = & gcloud @Arguments 2>&1; $code = $LASTEXITCODE }
    finally { $ErrorActionPreference = $old }
    if ($code -ne 0) { throw "gcloud failed ($code): gcloud $($Arguments -join ' ')`n$(@($output) -join [Environment]::NewLine)" }
    return @($output)
}

function Test-GcsObject {
    param([Parameter(Mandatory = $true)][string]$Uri)
    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { $output = & gcloud storage objects describe $Uri --format=json 2>&1; $code = $LASTEXITCODE }
    finally { $ErrorActionPreference = $old }
    if ($code -eq 0) { return $true }
    $message = @($output) -join [Environment]::NewLine
    if ($message -match '(?i)not found|does not exist|No URLs matched|matched no objects|404') { return $false }
    throw "Unable to determine GCS object state: $Uri`n$message"
}

function Get-GcsJson {
    param([Parameter(Mandatory = $true)][string]$Uri)
    $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    try { $raw = & gcloud storage cat $Uri 2>&1; $code = $LASTEXITCODE }
    finally { $ErrorActionPreference = $old }
    if ($code -ne 0) { throw "Unable to read existing immutable DONE: $Uri`n$(@($raw) -join [Environment]::NewLine)" }
    return (@($raw) -join "`n") | ConvertFrom-Json
}

function Get-InstancesByName {
    param([Parameter(Mandatory = $true)][string]$Name)
    $raw = Invoke-Gcloud @("compute", "instances", "list", "--project", $ProjectId, "--filter", ("name={0}" -f $Name), "--format=json")
    if (-not $raw) { return @() }
    return @(((@($raw) -join "`n") | ConvertFrom-Json))
}

function Convert-ToVmPrefix {
    param([Parameter(Mandatory = $true)][string]$Value)
    $name = ($Value.ToLowerInvariant() -replace '[^a-z0-9-]', '-').Trim('-')
    if (-not $name) { throw "RunName does not produce a VM prefix" }
    if ($name -notmatch '^[a-z]') { $name = "m43-$name" }
    if ($name.Length -gt 53) {
        $name = $name.Substring(0, 44).TrimEnd('-') + "-" + (Get-TextSha256 $Value).Substring(0, 8)
    }
    return $name
}

function Get-StrictInteger {
    param([Parameter(Mandatory = $true)]$Value, [Parameter(Mandatory = $true)][string]$Name)
    $integerTypes = @([byte], [sbyte], [int16], [uint16], [int32], [uint32], [int64], [uint64])
    if ($null -eq $Value -or $Value -is [bool] -or $integerTypes -notcontains $Value.GetType()) {
        throw "$Name must be a JSON integer, not bool/string/float"
    }
    return [int64]$Value
}

function Get-ObjectPropertyNames {
    param([Parameter(Mandatory = $true)]$Value)
    if ($Value -is [Collections.IDictionary]) {
        return @($Value.Keys | ForEach-Object { [string]$_ })
    }
    return @($Value.PSObject.Properties.Name)
}

function New-LocalSealedRebind {
    param(
        [Parameter(Mandatory = $true)][string[]]$TrainPaths,
        [Parameter(Mandatory = $true)][string[]]$CalibrationPaths,
        [Parameter(Mandatory = $true)][string]$DataContractPath,
        [Parameter(Mandatory = $true)][string]$PlanPath,
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [Parameter(Mandatory = $true)][string]$PredeclaredReceiptPath,
        [Parameter(Mandatory = $true)][string]$CloudContractPath,
        [Parameter(Mandatory = $true)][string]$RunManifestPath,
        [Parameter(Mandatory = $true)][string]$SourceArchivePath,
        [Parameter(Mandatory = $true)][string]$OutputPath
    )
    $rebindArguments = @("-m", "ofc_regular.train_hu_m43_fold_job", "rebind", "--run-name", $RunName)
    foreach ($path in $TrainPaths) { $rebindArguments += @("--train", $path) }
    foreach ($path in $CalibrationPaths) { $rebindArguments += @("--calibration", $path) }
    $rebindArguments += @(
        "--m43-data-contract", $DataContractPath, "--m43-plan", $PlanPath,
        "--repo-root", $RepoRoot, "--predeclared-receipt", $PredeclaredReceiptPath,
        "--fold-cloud-contract", $CloudContractPath, "--run-manifest", $RunManifestPath,
        "--source-archive", $SourceArchivePath, "--output-rebind-manifest", $OutputPath,
        "--cross-fit-folds", "$OuterFolds", "--iterations", "$Iterations",
        "--max-leaf-nodes", "$MaxLeafNodes", "--learning-rate", "$LearningRate",
        "--l2-regularization", "$L2Regularization", "--seed", "$Seed",
        "--paired-se-floor", "$PairedSeFloor", "--paired-huber-alpha", "$PairedHuberAlpha",
        "--downside-quantile", "$DownsideQuantile",
        "--positive-gain-score-weight", "$PositiveGainScoreWeight",
        "--downside-risk-score-weight", "$DownsideRiskScoreWeight",
        "--ensemble-disagreement-score-weight", "$EnsembleDisagreementScoreWeight",
        "--action-score-mode", $ActionScoreMode, "--model-id", $ModelId,
        "--near-best-margin", "$NearBestMargin", "--minimum-safe-teacher-gain", "$MinimumSafeTeacherGain",
        "--safety-calibrator-c", "$SafetyCalibratorC", "--safety-fit-ratio", "$SafetyFitRatio",
        "--safety-split-seed", "$SafetySplitSeed", "--minimum-safety-fit-samples", "$MinimumSafetyFitSamples",
        "--minimum-threshold-lock-samples", "$MinimumThresholdLockSamples", "--thresholds", $Thresholds,
        "--minimum-calibration-fires", "$MinimumCalibrationFires", "--maximum-false-positive-rate", "$MaximumFalsePositiveRate",
        "--maximum-p95-loss", "$MaximumP95Loss", "--maximum-p99-loss", "$MaximumP99Loss", "--maximum-max-loss", "$MaximumMaxLoss"
    )
    $oldPythonPath = $env:PYTHONPATH
    $oldPreference = $ErrorActionPreference
    $env:PYTHONPATH = Join-Path $RepoRoot "src"
    $ErrorActionPreference = "Continue"
    try { $output = @(& python @rebindArguments 2>&1); $code = $LASTEXITCODE }
    finally { $env:PYTHONPATH = $oldPythonPath; $ErrorActionPreference = $oldPreference }
    if ($code -ne 0) { throw "M4.3 local-only sealed rebind failed: $($output -join [Environment]::NewLine)" }
    if (-not (Test-Path -LiteralPath $OutputPath -PathType Leaf)) { throw "M4.3 local-only sealed rebind was not created" }
}

function Assert-SourcePackagePolicy {
    param(
        [Parameter(Mandatory = $true)]$Manifest,
        [Parameter(Mandatory = $true)][string]$ArchivePath
    )
    $policy = $Manifest.source_package_policy
    $expectedRoots = @("src/ofc_regular/train_hu_m43_fold_job.py", "src/ofc_regular/assemble_hu_m43_fold_training.py")
    $expectedPatterns = @("(?i)l[o]cked", "(?i)\.jsonl?$", "(?i)(^|/)(outputs?|configs?|data)(/|$)")
    if ([string]$policy.mode -ne "ast_recursive_local_import_closure_v1" -or
        (@($policy.roots) -join "|") -ne ($expectedRoots -join "|") -or
        (@($policy.forbidden_entry_patterns) -join "|") -ne ($expectedPatterns -join "|") -or
        $policy.generic_guard_literals_non_secret -ne $true -or $policy.sealed_paths_content_hashes_embedded -ne $false -or
        (Get-StrictInteger $policy.module_count "source_package_policy.module_count") -lt 3 -or
        [string]$policy.entries_sha256 -notmatch '^[0-9a-f]{64}$') {
        throw "Frozen source package policy changed"
    }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [IO.Compression.ZipFile]::OpenRead($ArchivePath)
    try {
        if (@($archive.Entries | Where-Object { $_.FullName -match '\\' }).Count -ne 0) { throw "Frozen source archive contains non-portable entry separators" }
        [string[]]$entries = @($archive.Entries | Where-Object { $_.FullName -match '\.py$' } | ForEach-Object { $_.FullName })
        [Array]::Sort($entries, [StringComparer]::Ordinal)
        $badEntries = @($entries | Where-Object { $_ -match '(?i)l[o]cked|\.jsonl?$|(^|/)(outputs?|configs?|data)(/|$)' })
        if ($badEntries.Count -ne 0 -or $entries.Count -ne (Get-StrictInteger $policy.module_count "source_package_policy.module_count")) {
            throw "Frozen source package entry coverage is invalid"
        }
        $canonicalEntries = '["' + ($entries -join '","') + '"]'
        if ((Get-TextSha256 $canonicalEntries) -ne [string]$policy.entries_sha256) {
            throw "Frozen source package entry digest mismatch"
        }
    }
    finally { $archive.Dispose() }
}

function Assert-FrozenManifest {
    param([Parameter(Mandatory = $true)]$Manifest)
    if ($Manifest.schema -ne "hu_m43_fold_spot_run_manifest_v1" -or $Manifest.status -ne "frozen" -or
        $Manifest.run_name -ne $RunName -or $Manifest.project_id -ne $ProjectId -or
        $Manifest.bucket -ne $Bucket -or $Manifest.current_profile_mutated -ne $false -or
        $Manifest.no_runtime_activation -ne $true -or (Get-StrictInteger $Manifest.job_count "job_count") -ne $ExpectedJobs -or
        (Get-StrictInteger $Manifest.compute.shard_count "compute.shard_count") -ne $ExpectedShards -or
        (Get-StrictInteger $Manifest.compute.jobs_per_shard "compute.jobs_per_shard") -ne $JobsPerShard) {
        throw "Frozen M4.3 fold run identity/schema is invalid"
    }
    if ([string]$Manifest.source.uri -ne "gs://$Bucket/runs/$RunName/source/ofc_regular_hu_m43_fold_source.zip" -or
        (Get-StrictInteger $Manifest.source.bytes "source.bytes") -lt 1 -or [string]$Manifest.source.sha256 -notmatch '^[0-9a-f]{64}$' -or
        [string]$Manifest.source.fold_worker_sha256 -notmatch '^[0-9a-f]{64}$' -or [string]$Manifest.source.assembler_sha256 -notmatch '^[0-9a-f]{64}$' -or
        [string]$Manifest.cloud_contract.uri -ne "gs://$Bucket/runs/$RunName/inputs/fold_cloud_contract.json" -or
        (Get-StrictInteger $Manifest.cloud_contract.bytes "cloud_contract.bytes") -lt 1 -or
        [string]$Manifest.startup.uri -ne "gs://$Bucket/runs/$RunName/source/startup_hu_m43_fold_group.sh" -or
        [string]$Manifest.startup.sha256 -notmatch '^[0-9a-f]{64}$' -or
        [string]$Manifest.dependencies.numpy -ne "2.2.6" -or [string]$Manifest.dependencies.scikit_learn -ne "1.8.0") {
        throw "Frozen M4.3 source/dependency transport binding is invalid"
    }
    foreach ($name in @("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")) {
        if ([string]$Manifest.process_environment.$name -ne "1") { throw "Frozen deterministic environment changed: $name" }
    }
    if ([string]$Manifest.process_environment.PYTHONHASHSEED -ne "0") { throw "Frozen PYTHONHASHSEED changed" }
    $sourcePolicy = $Manifest.source_package_policy
    $expectedRoots = @("src/ofc_regular/train_hu_m43_fold_job.py", "src/ofc_regular/assemble_hu_m43_fold_training.py")
    $expectedPatterns = @("(?i)l[o]cked", "(?i)\.jsonl?$", "(?i)(^|/)(outputs?|configs?|data)(/|$)")
    if ([string]$sourcePolicy.mode -ne "ast_recursive_local_import_closure_v1" -or
        (@($sourcePolicy.roots) -join "|") -ne ($expectedRoots -join "|") -or
        (@($sourcePolicy.forbidden_entry_patterns) -join "|") -ne ($expectedPatterns -join "|") -or
        (Get-StrictInteger $sourcePolicy.module_count "source_package_policy.module_count") -lt 3 -or
        [string]$sourcePolicy.entries_sha256 -notmatch '^[0-9a-f]{64}$' -or
        $sourcePolicy.generic_guard_literals_non_secret -ne $true -or $sourcePolicy.sealed_paths_content_hashes_embedded -ne $false) {
        throw "Frozen source package policy is invalid"
    }
    $cloudKeys = @(Get-ObjectPropertyNames $Manifest.cloud_contract | Sort-Object)
    $expectedCloudKeys = @(@("bytes", "contract_sha256", "file_sha256", "fold_plan_sha256", "predeclared_training_receipt_sha256", "train_identity_sha256", "uri") | Sort-Object)
    if (($cloudKeys -join "|") -ne ($expectedCloudKeys -join "|")) { throw "Frozen cloud binding key set changed" }
    $jobs = @($Manifest.jobs)
    if ($jobs.Count -ne $ExpectedJobs) { throw "Frozen M4.3 fold run must contain exactly 30 jobs" }
    $seen = [Collections.Generic.HashSet[int]]::new()
    for ($index = 0; $index -lt $ExpectedJobs; $index++) {
        $job = $jobs[$index]
        $slot = $index % 6
        $expectedKind = if ($slot -eq 0) { "outer_runtime" } else { "inner_oof_safety" }
        $expectedOuter = [int][Math]::Floor($index / 6)
        $expectedInner = if ($slot -eq 0) { $null } else { $slot - 1 }
        $jobIndex = Get-StrictInteger $job.job_index "jobs[$index].job_index"
        $outerFold = Get-StrictInteger $job.outer_fold "jobs[$index].outer_fold"
        $shardIndex = Get-StrictInteger $job.shard_index "jobs[$index].shard_index"
        $innerFold = if ($null -eq $job.inner_fold) { $null } else { Get-StrictInteger $job.inner_fold "jobs[$index].inner_fold" }
        $jobSpec = $job.job_spec
        $specInner = if ($null -eq $jobSpec.inner_fold) { $null } else { Get-StrictInteger $jobSpec.inner_fold "jobs[$index].job_spec.inner_fold" }
        foreach ($field in @("job_index", "outer_fold", "estimator_fold_index", "estimator_seed", "fit_samples", "outer_validation_samples", "inner_validation_samples")) {
            Get-StrictInteger $jobSpec.$field "jobs[$index].job_spec.$field" | Out-Null
        }
        if (-not $seen.Add([int]$jobIndex) -or $jobIndex -ne $index -or
            [string]$job.job_kind -ne $expectedKind -or
            $outerFold -ne $expectedOuter -or $shardIndex -ne [int][Math]::Floor($index / $JobsPerShard) -or
            (($null -eq $expectedInner -and $null -ne $job.inner_fold) -or
             ($null -ne $expectedInner -and $innerFold -ne $expectedInner)) -or
            $null -eq $jobSpec -or (Get-StrictInteger $jobSpec.job_index "job_spec.job_index") -ne $index -or
            [string]$jobSpec.kind -ne $expectedKind -or (Get-StrictInteger $jobSpec.outer_fold "job_spec.outer_fold") -ne $expectedOuter -or
            (($null -eq $expectedInner -and $null -ne $jobSpec.inner_fold) -or ($null -ne $expectedInner -and $specInner -ne $expectedInner)) -or
            [string]$job.job_spec_sha256 -notmatch '^[0-9a-f]{64}$' -or
            [string]$job.done_uri -ne ("gs://$Bucket/runs/$RunName/results/job-{0:D2}/DONE.json" -f $index)) {
            throw "Frozen M4.3 job mapping is invalid at index $index"
        }
    }
    if (@(Get-ObjectPropertyNames $Manifest.inputs | Where-Object { $_ -notin @("train", "calibration", "input_bundle_sha256") }).Count -ne 0) {
        throw "Frozen cloud manifest contains an unauthorized input split"
    }
    foreach ($split in @("train", "calibration")) {
        $entries = @($Manifest.inputs.$split)
        for ($i = 0; $i -lt $entries.Count; $i++) {
            if ((Get-StrictInteger $entries[$i].index "inputs.$split[$i].index") -ne $i -or
                (Get-StrictInteger $entries[$i].bytes "inputs.$split[$i].bytes") -lt 1 -or
                (Get-StrictInteger $entries[$i].rows "inputs.$split[$i].rows") -lt 1) {
                throw "Frozen input integer provenance is invalid for $split index $i"
            }
        }
    }
    $hp = $Manifest.training_config
    if ((Get-StrictInteger $hp.cross_fit_folds "training_config.cross_fit_folds") -ne 5 -or
        (Get-StrictInteger $hp.iterations "training_config.iterations") -ne 150 -or
        (Get-StrictInteger $hp.max_leaf_nodes "training_config.max_leaf_nodes") -ne 31 -or
        (Get-StrictInteger $hp.seed "training_config.seed") -ne 2026071801 -or
        (Get-StrictInteger $hp.safety_split_seed "training_config.safety_split_seed") -ne 2026071802 -or
        (Get-StrictInteger $hp.minimum_safety_fit_samples "training_config.minimum_safety_fit_samples") -ne 30 -or
        (Get-StrictInteger $hp.minimum_threshold_lock_samples "training_config.minimum_threshold_lock_samples") -ne 30 -or
        (Get-StrictInteger $hp.minimum_calibration_fires "training_config.minimum_calibration_fires") -ne 10 -or
        [string]$hp.action_score_mode -ne "baseline_paired_delta_risk_ensemble_v3" -or
        [string]$hp.model_id -ne "hu-m43-t1-second-regular-hu-m43-c2e16-pilot200-20260713-1810" -or
        [double]$hp.positive_gain_score_weight -ne 0.25 -or [double]$hp.downside_risk_score_weight -ne 0.5 -or
        [double]$hp.ensemble_disagreement_score_weight -ne 0.25 -or [string]$Manifest.training_config_sha256 -notmatch '^[0-9a-f]{64}$') {
        throw "Frozen full M4.3 training configuration changed"
    }
}

function Assert-FrozenDone {
    param(
        [Parameter(Mandatory = $true)]$Done,
        [Parameter(Mandatory = $true)]$ExpectedJob,
        [Parameter(Mandatory = $true)][int]$ExpectedIndex,
        [Parameter(Mandatory = $true)]$Manifest,
        [Parameter(Mandatory = $true)][string]$ManifestSha
    )
    $doneIndex = Get-StrictInteger $Done.job_index "DONE.job_index"
    $doneOuter = Get-StrictInteger $Done.outer_fold "DONE.outer_fold"
    $doneInner = if ($null -eq $Done.inner_fold) { $null } else { Get-StrictInteger $Done.inner_fold "DONE.inner_fold" }
    if ($Done.schema -ne "hu_m43_fold_job_done_v1" -or $Done.status -ne "complete" -or
        $Done.run_name -ne $RunName -or $doneIndex -ne $ExpectedIndex -or
        [string]$Done.job_kind -ne [string]$ExpectedJob.job_kind -or $doneOuter -ne (Get-StrictInteger $ExpectedJob.outer_fold "manifest.outer_fold") -or
        (($null -eq $ExpectedJob.inner_fold -and $null -ne $Done.inner_fold) -or
         ($null -ne $ExpectedJob.inner_fold -and $doneInner -ne (Get-StrictInteger $ExpectedJob.inner_fold "manifest.inner_fold"))) -or
        [string]$Done.job_spec_sha256 -ne [string]$ExpectedJob.job_spec_sha256 -or
        [string]$Done.source_sha256 -ne [string]$Manifest.source.sha256 -or
        [string]$Done.run_manifest_sha256 -ne $ManifestSha -or
        [string]$Done.cloud_contract_sha256 -ne [string]$Manifest.cloud_contract.file_sha256 -or
        [string]$Done.input_bundle_sha256 -ne [string]$Manifest.inputs.input_bundle_sha256 -or
        [string]$Done.artifact_sha256 -notmatch '^[0-9a-f]{64}$' -or
        [string]$Done.job_manifest_sha256 -notmatch '^[0-9a-f]{64}$' -or
        $Done.current_profile_mutated -ne $false -or $Done.no_runtime_activation -ne $true) {
        throw "Existing immutable DONE is stale/invalid for job $ExpectedIndex"
    }
}

if ($RunName -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$') { throw "RunName is not path-safe" }
if ($BootDiskGb -lt 20) { throw "BootDiskGb must be at least 20" }
if (-not $Zones -or $Zones.Count -eq 0) { throw "At least one zone is required" }
if ($ResumeExisting -and -not $CreateInstances) { throw "ResumeExisting requires CreateInstances" }
if ($ResumeExisting -and $DryRun) { throw "ResumeExisting and DryRun are mutually exclusive" }
$uniqueShards = @($StartShards | Sort-Object -Unique)
if ($uniqueShards.Count -ne $StartShards.Count -or @($uniqueShards | Where-Object { $_ -lt 0 -or $_ -ge $ExpectedShards }).Count -ne 0) {
    throw "StartShards must be unique values in 0..7"
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runDir = Join-Path $repoRoot "outputs/gcp_runs/$RunName"
$artifactDir = if ($DryRun) { Join-Path $runDir "dryrun" } else { $runDir }
$manifestPath = Join-Path $artifactDir "m43_fold_run_manifest.json"
$startupPath = Join-Path $artifactDir "startup_hu_m43_fold_group.sh"
$sourcePath = Join-Path $artifactDir "ofc_regular_hu_m43_fold_source.zip"
$cloudContractPath = Join-Path $artifactDir "fold_cloud_contract.json"
$localRebindPath = Join-Path $artifactDir "m43_local_sealed_rebind_manifest.json"
$dryRunPlanPath = Join-Path $artifactDir "dryrun_plan.json"
$gcsPrefix = "gs://$Bucket/runs/$RunName"
$manifestUri = "$gcsPrefix/source/m43_fold_run_manifest.json"
$startupUri = "$gcsPrefix/source/startup_hu_m43_fold_group.sh"
$sourceUri = "$gcsPrefix/source/ofc_regular_hu_m43_fold_source.zip"
$cloudContractUri = "$gcsPrefix/inputs/fold_cloud_contract.json"
$vmPrefix = Convert-ToVmPrefix $RunName

if ($DryRun -and (Test-Path -LiteralPath $runDir)) {
    throw "DryRun RunName is immutable and already exists; refusing to pollute or overwrite: $runDir"
}

$remoteManifestExists = $false
if (-not $DryRun) { $remoteManifestExists = Test-GcsObject $manifestUri }
if ($ResumeExisting -and -not $remoteManifestExists) { throw "Frozen remote M4.3 fold run does not exist: $manifestUri" }
if (-not $ResumeExisting -and $remoteManifestExists) { throw "M4.3 fold run already exists and is immutable; use ResumeExisting" }

if ($ResumeExisting) {
    if (-not $Train -or -not $Calibration -or -not $M43DataContract) {
        throw "ResumeExisting requires Train, Calibration, and M43DataContract to verify the local-only sealed rebind"
    }
    if (-not (Test-Path -LiteralPath $localRebindPath -PathType Leaf)) {
        throw "ResumeExisting requires the original local-only sealed rebind manifest: $localRebindPath"
    }
    $resolvedTrain = @($Train | ForEach-Object { Resolve-InputPath $_ $repoRoot })
    $resolvedCalibration = @($Calibration | ForEach-Object { Resolve-InputPath $_ $repoRoot })
    $resolvedContract = Resolve-InputPath $M43DataContract $repoRoot
    $resolvedPlan = Resolve-InputPath $M43Plan $repoRoot
    $predeclaredReceipt = Resolve-InputPath "outputs/hu_joint_policy/m43_spot/regular-hu-m43-c2e16-pilot200-20260713-1810/m43_t1_second_attempt01/local_training_aborted.json" $repoRoot
    $temp = Join-Path ([IO.Path]::GetTempPath()) ("m43-fold-resume-" + [guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $temp | Out-Null
    try {
        $remoteManifestPath = Join-Path $temp "manifest.json"
        $remoteStartupPath = Join-Path $temp "startup.sh"
        $remoteContractPath = Join-Path $temp "fold_cloud_contract.json"
        $remoteSourcePath = Join-Path $temp "source.zip"
        Invoke-Gcloud @("storage", "cp", $manifestUri, $remoteManifestPath, "--project", $ProjectId) | Out-Null
        $manifest = Get-Content -LiteralPath $remoteManifestPath -Raw | ConvertFrom-Json
        Assert-FrozenManifest $manifest
        Invoke-Gcloud @("storage", "cp", $startupUri, $remoteStartupPath, "--project", $ProjectId) | Out-Null
        Invoke-Gcloud @("storage", "cp", $cloudContractUri, $remoteContractPath, "--project", $ProjectId) | Out-Null
        Invoke-Gcloud @("storage", "cp", "$($manifest.source.uri)", $remoteSourcePath, "--project", $ProjectId) | Out-Null
        if ((Get-Sha256 $remoteStartupPath) -ne [string]$manifest.startup.sha256 -or
            (Get-Sha256 $remoteContractPath) -ne [string]$manifest.cloud_contract.file_sha256 -or
            (Get-Sha256 $remoteSourcePath) -ne [string]$manifest.source.sha256 -or
            [long](Get-Item -LiteralPath $remoteSourcePath).Length -ne (Get-StrictInteger $manifest.source.bytes "source.bytes")) {
            throw "Frozen remote startup, cloud contract, or source archive hash mismatch"
        }
        $remoteManifestRaw = Get-Content -LiteralPath $remoteManifestPath -Raw
        $remoteContractRaw = Get-Content -LiteralPath $remoteContractPath -Raw
        if ($remoteManifestRaw -match '(?i)locked' -or $remoteContractRaw -match '(?i)locked') {
            throw "Frozen cloud state contains a forbidden holdout token"
        }
        $bindingScript = @'
import hashlib,json,sys
m=json.load(open(sys.argv[1],encoding="utf-8")); p=json.load(open(sys.argv[2],encoding="utf-8"))
canonical=lambda value:hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()
unsigned=dict(p); declared=unsigned.pop("contract_sha256"); assert declared==canonical(unsigned)
assert m["training_config"]==p["hyperparameters"]
assert m["dependencies"]==p["dependencies"]=={"numpy":"2.2.6","scikit_learn":"1.8.0"}
assert m["process_environment"]==p["process_environment"]=={"PYTHONHASHSEED":"0","OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1"}
assert m["training_config_sha256"]==canonical(p["hyperparameters"])
assert m["inputs"]["input_bundle_sha256"]==p["input_bundle_sha256"]
assert m["cloud_contract"]["predeclared_training_receipt_sha256"]==p["predeclared_training_receipt_sha256"]=="9313bdc6ffa33335032ee3d98e279d123d2f3aa70571114e75ee5f38e2594bea"
for split in ("train","calibration"):
 assert len(m["inputs"][split])==len(p["inputs"][split])
 for actual,expected in zip(m["inputs"][split],p["inputs"][split]):
  stripped={k:v for k,v in actual.items() if k!="uri"}; assert stripped==expected and actual["uri"].endswith(f'{split}-{expected["index"]:03d}.jsonl')
assert len(m["jobs"])==len(p["fold_plan"]["jobs"])==30
for i,(actual,projected) in enumerate(zip(m["jobs"],p["fold_plan"]["jobs"])):
 spec={k:v for k,v in projected.items() if k!="job_spec_sha256"}
 assert actual["job_spec"]==spec and actual["job_spec_sha256"]==projected["job_spec_sha256"]
 assert type(actual["job_index"]) is int and actual["job_index"]==i and actual["job_kind"]==projected["kind"] and actual["outer_fold"]==projected["outer_fold"] and actual["inner_fold"]==projected["inner_fold"]
'@
        $oldPreference = $ErrorActionPreference; $ErrorActionPreference = "Continue"
        try { $bindingOutput = @($bindingScript | & python - $remoteManifestPath $remoteContractPath 2>&1); $bindingCode = $LASTEXITCODE }
        finally { $ErrorActionPreference = $oldPreference }
        if ($bindingCode -ne 0) { throw "Frozen remote config/projection binding is invalid: $($bindingOutput -join [Environment]::NewLine)" }
        $manifestSha256 = Get-Sha256 $remoteManifestPath
        New-Item -ItemType Directory -Path $runDir -Force | Out-Null
        foreach ($copy in @(
            @($remoteManifestPath, $manifestPath),
            @($remoteStartupPath, $startupPath),
            @($remoteContractPath, $cloudContractPath),
            @($remoteSourcePath, $sourcePath)
        )) {
            if (Test-Path -LiteralPath $copy[1] -PathType Leaf) {
                if ((Get-Sha256 $copy[0]) -ne (Get-Sha256 $copy[1])) { throw "Local frozen run file differs: $($copy[1])" }
            }
            else { Write-BytesCreateNew -Path $copy[1] -Bytes ([IO.File]::ReadAllBytes($copy[0])) }
        }
        $candidateRebind = Join-Path $temp "candidate_local_rebind.json"
        New-LocalSealedRebind -TrainPaths $resolvedTrain -CalibrationPaths $resolvedCalibration `
            -DataContractPath $resolvedContract -PlanPath $resolvedPlan -RepoRoot $repoRoot `
            -PredeclaredReceiptPath $predeclaredReceipt -CloudContractPath $cloudContractPath `
            -RunManifestPath $manifestPath -SourceArchivePath $sourcePath -OutputPath $candidateRebind
        if ((Get-Sha256 $candidateRebind) -ne (Get-Sha256 $localRebindPath)) {
            throw "Original local-only sealed rebind is stale or mismatched"
        }
    }
    finally { Remove-Item -LiteralPath $temp -Recurse -Force -ErrorAction SilentlyContinue }
}
else {
    if (-not $Train -or -not $Calibration -or -not $M43DataContract) {
        throw "New runs require Train, Calibration, and M43DataContract"
    }
    foreach ($path in @($manifestPath, $startupPath, $sourcePath, $cloudContractPath, $localRebindPath)) {
        if (Test-Path -LiteralPath $path) { throw "Local frozen run artifact already exists: $path" }
    }
    $resolvedTrain = @($Train | ForEach-Object { Resolve-InputPath $_ $repoRoot })
    $resolvedCalibration = @($Calibration | ForEach-Object { Resolve-InputPath $_ $repoRoot })
    $resolvedContract = Resolve-InputPath $M43DataContract $repoRoot
    $resolvedPlan = Resolve-InputPath $M43Plan $repoRoot
    $predeclaredReceipt = Resolve-InputPath "outputs/hu_joint_policy/m43_spot/regular-hu-m43-c2e16-pilot200-20260713-1810/m43_t1_second_attempt01/local_training_aborted.json" $repoRoot
    if ((Get-Sha256 $predeclaredReceipt) -ne $ExpectedPredeclaredReceiptSha256) {
        throw "Predeclared attempt receipt SHA-256 changed"
    }
    $foldWorkerPath = Resolve-InputPath "src/ofc_regular/train_hu_m43_fold_job.py" $repoRoot
    $assemblerPath = Resolve-InputPath "src/ofc_regular/assemble_hu_m43_fold_training.py" $repoRoot

    $temp = Join-Path ([IO.Path]::GetTempPath()) ("m43-fold-freeze-" + [guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $temp | Out-Null
    try {
        $preparedContract = Join-Path $temp "fold_cloud_contract.json"
        $prepareArgs = @("-m", "ofc_regular.train_hu_m43_fold_job", "prepare")
        foreach ($path in $resolvedTrain) { $prepareArgs += @("--train", $path) }
        foreach ($path in $resolvedCalibration) { $prepareArgs += @("--calibration", $path) }
        $prepareArgs += @(
            "--m43-data-contract", $resolvedContract, "--m43-plan", $resolvedPlan,
            "--repo-root", $repoRoot, "--predeclared-receipt", $predeclaredReceipt, "--output-contract", $preparedContract,
            "--cross-fit-folds", "$OuterFolds", "--iterations", "$Iterations",
            "--max-leaf-nodes", "$MaxLeafNodes", "--learning-rate", "$LearningRate",
            "--l2-regularization", "$L2Regularization", "--seed", "$Seed",
            "--paired-se-floor", "$PairedSeFloor", "--paired-huber-alpha", "$PairedHuberAlpha",
            "--downside-quantile", "$DownsideQuantile",
            "--positive-gain-score-weight", "$PositiveGainScoreWeight",
            "--downside-risk-score-weight", "$DownsideRiskScoreWeight",
            "--ensemble-disagreement-score-weight", "$EnsembleDisagreementScoreWeight",
            "--action-score-mode", $ActionScoreMode, "--model-id", $ModelId,
            "--near-best-margin", "$NearBestMargin", "--minimum-safe-teacher-gain", "$MinimumSafeTeacherGain",
            "--safety-calibrator-c", "$SafetyCalibratorC", "--safety-fit-ratio", "$SafetyFitRatio",
            "--safety-split-seed", "$SafetySplitSeed", "--minimum-safety-fit-samples", "$MinimumSafetyFitSamples",
            "--minimum-threshold-lock-samples", "$MinimumThresholdLockSamples", "--thresholds", $Thresholds,
            "--minimum-calibration-fires", "$MinimumCalibrationFires", "--maximum-false-positive-rate", "$MaximumFalsePositiveRate",
            "--maximum-p95-loss", "$MaximumP95Loss", "--maximum-p99-loss", "$MaximumP99Loss", "--maximum-max-loss", "$MaximumMaxLoss"
        )
        $oldPythonPath = $env:PYTHONPATH
        $env:PYTHONPATH = Join-Path $repoRoot "src"
        $oldPreference = $ErrorActionPreference; $ErrorActionPreference = "Continue"
        try { $prepareOutput = @(& python @prepareArgs 2>&1); $prepareCode = $LASTEXITCODE }
        finally { $ErrorActionPreference = $oldPreference; $env:PYTHONPATH = $oldPythonPath }
        if ($prepareCode -ne 0) { throw "M4.3 cloud-contract preparation failed: $($prepareOutput -join [Environment]::NewLine)" }

        $validationScript = @'
import hashlib,json,sys
projection_path,full_path,plan_path,predeclared_path,*paths=sys.argv[1:]
train_count=int(paths.pop(0)); train=paths[:train_count]; paths=paths[train_count:]
cal_count=int(paths.pop(0)); cal=paths[:cal_count]; paths=paths[cal_count:]; assert not paths
h=lambda p: hashlib.sha256(open(p,"rb").read()).hexdigest()
def canonical_without_self(obj):
    copy=dict(obj); copy.pop("contract_sha256",None)
    return hashlib.sha256(json.dumps(copy,sort_keys=True,separators=(",",":")).encode()).hexdigest()
p=json.load(open(projection_path,encoding="utf-8")); full=json.load(open(full_path,encoding="utf-8"))
def scan(value):
    if isinstance(value,dict):
        for k,v in value.items():
            assert "locked" not in str(k).lower(); scan(v)
    elif isinstance(value,list):
        for v in value: scan(v)
    elif isinstance(value,str): assert "locked" not in value.lower()
scan(p)
assert p["schema"]=="hu_m43_fold_cloud_contract_v1" and p["status"]=="frozen_cloud_safe" and p["cloud_safe"] is True
assert p["dependencies"]=={"numpy":"2.2.6","scikit_learn":"1.8.0"}
assert p["process_environment"]=={"PYTHONHASHSEED":"0","OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1"}
assert p["contract_sha256"]==canonical_without_self(p)
assert p["predeclared_training_receipt_sha256"]==h(predeclared_path)=="9313bdc6ffa33335032ee3d98e279d123d2f3aa70571114e75ee5f38e2594bea"
for key,actual in (("train",train),("calibration",cal)):
    entries=p["inputs"][key]; assert len(entries)==len(actual)
    for i,(entry,path) in enumerate(zip(entries,actual)):
        assert entry["index"]==i and entry["sha256"]==h(path) and entry["bytes"]==len(open(path,"rb").read())
f=p["fold_plan"]
assert f["outer_folds"]==5 and f["inner_folds_per_outer"]==5 and f["total_jobs"]==30
for key in ("train_identity_sha256","fold_plan_sha256"):
    assert len(f[key])==64 and int(f[key],16)>=0
hp=p["hyperparameters"]
expected_hp={"cross_fit_folds":5,"iterations":150,"max_leaf_nodes":31,"learning_rate":0.05,"seed":2026071801,"paired_se_floor":0.5,"paired_huber_alpha":0.9,"downside_quantile":0.9,"positive_gain_score_weight":0.25,"downside_risk_score_weight":0.5,"ensemble_disagreement_score_weight":0.25,"action_score_mode":"baseline_paired_delta_risk_ensemble_v3","model_id":"hu-m43-t1-second-regular-hu-m43-c2e16-pilot200-20260713-1810","near_best_margin":0.5,"minimum_safe_teacher_gain":0.0,"l2_regularization":1.0,"safety_calibrator_c":0.25,"safety_fit_ratio":0.5,"safety_split_seed":2026071802,"minimum_safety_fit_samples":30,"minimum_threshold_lock_samples":30,"thresholds":[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.975,0.99,1.0],"minimum_calibration_fires":10,"maximum_false_positive_rate":0.3,"maximum_p95_loss":25.0,"maximum_p99_loss":40.0,"maximum_max_loss":50.0}
assert hp==expected_hp
jobs=f["jobs"]; assert len(jobs)==30
for i,j in enumerate(jobs):
    assert type(j["job_index"]) is int and j["job_index"]==i
    assert j["outer_fold"]==i//6 and j["kind"]==("outer_runtime" if i%6==0 else "inner_oof_safety")
    assert j["inner_fold"]==(None if i%6==0 else i%6-1) and len(j["job_spec_sha256"])==64
canonical=lambda obj:hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(",",":")).encode()).hexdigest()
print(json.dumps({"file_sha256":h(projection_path),"contract_sha256":p["contract_sha256"],"predeclared_sha256":p["predeclared_training_receipt_sha256"],"input_bundle_sha256":p["input_bundle_sha256"],"hyperparameters_sha256":canonical(hp),"train_identity_sha256":f["train_identity_sha256"],"fold_plan_sha256":f["fold_plan_sha256"]}))
'@
        $validatorArgs = @($preparedContract, $resolvedContract, $resolvedPlan, $predeclaredReceipt, "$($resolvedTrain.Count)") + $resolvedTrain + @("$($resolvedCalibration.Count)") + $resolvedCalibration
        $oldPreference = $ErrorActionPreference; $ErrorActionPreference = "Continue"
        try { $projectionSummaryRaw = @($validationScript | & python - @validatorArgs 2>&1); $projectionCode = $LASTEXITCODE }
        finally { $ErrorActionPreference = $oldPreference }
        if ($projectionCode -ne 0) { throw "Prepared fold cloud contract is not fail-closed: $($projectionSummaryRaw -join [Environment]::NewLine)" }
        $projectionSummary = (@($projectionSummaryRaw) -join "`n") | ConvertFrom-Json

        $packageRoot = Join-Path $temp "package"
        $sourceClosureBuilder = @'
import ast,json,pathlib,shutil,sys
repo=pathlib.Path(sys.argv[1]).resolve(); destination=pathlib.Path(sys.argv[2]).resolve()
source=repo/"src"/"ofc_regular"
queue=["ofc_regular", "ofc_regular.train_hu_m43_fold_job", "ofc_regular.assemble_hu_m43_fold_training"]
seen=set(); selected=[]
def module_path(name):
    parts=name.split(".")
    if parts[0]!="ofc_regular": raise ValueError(f"outside package: {name}")
    candidate=source.joinpath(*parts[1:])
    file=candidate/"__init__.py" if candidate.is_dir() else candidate.with_suffix(".py")
    return file if file.is_file() else None
def add(name):
    if name.startswith("ofc_regular") and name not in seen and module_path(name) is not None: queue.append(name)
while queue:
    name=queue.pop(0)
    if name in seen: continue
    path=module_path(name)
    if path is None: raise ValueError(f"missing local module: {name}")
    seen.add(name); selected.append((name,path))
    tree=ast.parse(path.read_text(encoding="utf-8"),filename=str(path))
    package=name if path.name=="__init__.py" else name.rsplit(".",1)[0]
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):
            for alias in node.names: add(alias.name)
        elif isinstance(node,ast.ImportFrom):
            if node.level:
                base=package.split(".")
                if node.level>1: base=base[:-(node.level-1)]
                target=".".join(base+(([node.module] if node.module else [])))
                add(target)
                if node.module is None:
                    for alias in node.names: add(target+"."+alias.name)
            elif node.module: add(node.module)
entries=[]
for name,path in selected:
    relative=path.relative_to(repo/"src")
    if any(token in relative.as_posix().lower() for token in ("locked","config","output","jsonl")):
        raise ValueError(f"forbidden dependency entry name: {relative}")
    target=destination/"src"/relative
    target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(path,target)
    entries.append("src/"+str(relative).replace("\\","/"))
entries=sorted(entries)
digest=__import__("hashlib").sha256(json.dumps(entries,separators=(",",":")).encode()).hexdigest()
print(json.dumps({"entries":entries,"entries_sha256":digest},sort_keys=True))
'@
        $oldPreference = $ErrorActionPreference; $ErrorActionPreference = "Continue"
        try { $closureOutput = @($sourceClosureBuilder | & python - $repoRoot $packageRoot 2>&1); $closureCode = $LASTEXITCODE }
        finally { $ErrorActionPreference = $oldPreference }
        if ($closureCode -ne 0) { throw "Unable to build minimal AST dependency closure: $($closureOutput -join [Environment]::NewLine)" }
        $sourceClosureInfo = (@($closureOutput) -join "`n") | ConvertFrom-Json
        $sourceClosure = @($sourceClosureInfo.entries)
        if ($sourceClosure.Count -lt 3 -or "src/ofc_regular/train_hu_m43_fold_job.py" -notin $sourceClosure -or
            "src/ofc_regular/assemble_hu_m43_fold_training.py" -notin $sourceClosure -or
            [string]$sourceClosureInfo.entries_sha256 -notmatch '^[0-9a-f]{64}$') {
            throw "Minimal source closure omitted a required M4.3 entrypoint"
        }
        $packageFiles = @(Get-ChildItem -LiteralPath (Join-Path $packageRoot "src/ofc_regular") -Recurse -File)
        $forbiddenSourceTokens = @($repoRoot, $resolvedContract, $resolvedPlan, $predeclaredReceipt, (Get-Sha256 $resolvedContract), (Get-Sha256 $resolvedPlan))
        $forbiddenSourceTokens += @($resolvedTrain) + @($resolvedCalibration)
        foreach ($file in $packageFiles) {
            $relative = $file.FullName.Substring($packageRoot.Length).TrimStart('\', '/') -replace '\\','/'
            if ($relative -match '(?i)locked|(^|/)(outputs?|configs?|data)(/|$)|\.jsonl?$') { throw "Source closure contains a forbidden entry name: $relative" }
            $content = Get-Content -LiteralPath $file.FullName -Raw
            foreach ($token in $forbiddenSourceTokens) {
                if ($token -and $content.IndexOf([string]$token, [StringComparison]::OrdinalIgnoreCase) -ge 0) {
                    throw "Source closure contains local/sealed path or digest material: $relative"
                }
            }
        }
        $tempSource = Join-Path $temp "source.zip"
        $zipBuilder = @'
import pathlib,sys,zipfile
source=pathlib.Path(sys.argv[1]); destination=pathlib.Path(sys.argv[2])
with zipfile.ZipFile(destination,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
    for path in sorted(source.rglob("*.py")):
        arcname=path.relative_to(source).as_posix()
        if "\\" in arcname: raise ValueError("non-portable archive separator")
        archive.write(path,arcname)
'@
        $oldPreference = $ErrorActionPreference; $ErrorActionPreference = "Continue"
        try { $zipOutput = @($zipBuilder | & python - $packageRoot $tempSource 2>&1); $zipCode = $LASTEXITCODE }
        finally { $ErrorActionPreference = $oldPreference }
        if ($zipCode -ne 0) { throw "Portable source archive creation failed: $($zipOutput -join [Environment]::NewLine)" }
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        $zip = [IO.Compression.ZipFile]::OpenRead($tempSource)
        try {
            $badEntries = @($zip.Entries | Where-Object { $_.FullName -match '\\|(?i)locked|\.jsonl?$|(^|/)(outputs?|configs?|data)(/|$)' })
            if ($badEntries.Count -ne 0) { throw "Source package contains a data/config artifact" }
        }
        finally { $zip.Dispose() }

        $cloudProjection = Get-Content -LiteralPath $preparedContract -Raw | ConvertFrom-Json
        $inputPlan = [ordered]@{ train = @(); calibration = @() }
        foreach ($split in @("train", "calibration")) {
            $paths = @(if ($split -eq "train") { $resolvedTrain } else { $resolvedCalibration })
            $projectedInputs = @($cloudProjection.inputs.$split)
            if ($paths.Count -ne $projectedInputs.Count) { throw "Projection input count changed for $split" }
            for ($i = 0; $i -lt $paths.Count; $i++) {
                $projected = $projectedInputs[$i]
                if ((Get-StrictInteger $projected.index "projection.inputs.$split[$i].index") -ne $i -or
                    [string]$projected.sha256 -ne (Get-Sha256 $paths[$i]) -or
                    (Get-StrictInteger $projected.bytes "projection.inputs.$split[$i].bytes") -ne [long](Get-Item -LiteralPath $paths[$i]).Length) {
                    throw "Projection input provenance changed for $split index $i"
                }
                $inputPlan[$split] += [ordered]@{
                    index = $i; bytes = [long]$projected.bytes; rows = Get-StrictInteger $projected.rows "projection.inputs.$split[$i].rows"
                    sha256 = [string]$projected.sha256
                    uri = ("$gcsPrefix/inputs/{0}-{1:D3}.jsonl" -f $split, $i)
                }
            }
        }
        $inputBundleSha256 = [string]$cloudProjection.input_bundle_sha256

        $startup = @'
#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
LOG=/var/log/hu_m43_fold_group.log
exec > >(tee -a "$LOG") 2>&1
META=http://metadata.google.internal/computeMetadata/v1
meta(){ curl -fsS -H 'Metadata-Flavor: Google' "$META/instance/attributes/$1"; }
project_meta(){ curl -fsS -H 'Metadata-Flavor: Google' "$META/project/project-id"; }
instance_meta(){ curl -fsS -H 'Metadata-Flavor: Google' "$META/instance/$1"; }
token(){ curl -fsS -H 'Metadata-Flavor: Google' "$META/instance/service-accounts/default/token" | python3 -c 'import json,sys;print(json.load(sys.stdin)["access_token"])'; }
urlencode(){ python3 -c 'import sys,urllib.parse;print(urllib.parse.quote(sys.argv[1],safe=""))' "$1"; }
PROJECT_ID="$(project_meta)"; BUCKET="$(meta BUCKET)"; RUN_NAME="$(meta RUN_NAME)"
PREFIX="runs/${RUN_NAME}"; MANIFEST_OBJECT="$(meta MANIFEST_OBJECT)"; MANIFEST_SHA256="$(meta MANIFEST_SHA256)"
JOB_INDICES="$(meta JOB_INDICES)"; INSTANCE_NAME="$(instance_meta name)"; ZONE="$(instance_meta zone)"; ZONE="${ZONE##*/}"
WORK=/work/hu_m43_fold; mkdir -p "$WORK"
gcs_download(){ local o="$1" d="$2" e a; e="$(urlencode "$o")"; a="$(token)"; curl --retry 5 --retry-all-errors -fsSL -H "Authorization: Bearer ${a}" "https://storage.googleapis.com/storage/v1/b/${BUCKET}/o/${e}?alt=media" -o "$d"; }
gcs_upload(){ local s="$1" o="$2" e a; [[ -f "$s" ]] || return 1; e="$(urlencode "$o")"; a="$(token)"; curl --retry 5 --retry-all-errors -fsS -X POST -H "Authorization: Bearer ${a}" -H 'Content-Type: application/octet-stream' --data-binary "@${s}" "https://storage.googleapis.com/upload/storage/v1/b/${BUCKET}/o?uploadType=media&name=${e}" >/dev/null; }
gcs_upload_immutable(){ local s="$1" o="$2" e a; [[ -f "$s" ]] || return 1; e="$(urlencode "$o")"; a="$(token)"; curl --retry 5 --retry-all-errors -fsS -X POST -H "Authorization: Bearer ${a}" -H 'Content-Type: application/octet-stream' --data-binary "@${s}" "https://storage.googleapis.com/upload/storage/v1/b/${BUCKET}/o?uploadType=media&name=${e}&ifGenerationMatch=0" >/dev/null; }
self_delete(){ local a; a="$(token)" || return 0; curl -fsS -X DELETE -H "Authorization: Bearer ${a}" "https://compute.googleapis.com/compute/v1/projects/${PROJECT_ID}/zones/${ZONE}/instances/${INSTANCE_NAME}" >/dev/null || true; }
trap 'self_delete' EXIT
apt-get update -y
apt-get install -y unzip curl ca-certificates python3 python3-venv python3-pip
gcs_download "$MANIFEST_OBJECT" "$WORK/run_manifest.json"
echo "${MANIFEST_SHA256}  $WORK/run_manifest.json" | sha256sum -c -
eval "$(python3 - "$WORK/run_manifest.json" <<'PY'
import json,shlex,sys
m=json.load(open(sys.argv[1],encoding="utf-8"))
assert m["schema"]=="hu_m43_fold_spot_run_manifest_v1" and m["status"]=="frozen" and m["job_count"]==30
assert set(m["inputs"])=={"train","calibration","input_bundle_sha256"}
assert m["dependencies"]=={"numpy":"2.2.6","scikit_learn":"1.8.0"}
assert m["process_environment"]=={"PYTHONHASHSEED":"0","OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1"}
def scan(x):
 forbidden="lo"+"cked"
 if isinstance(x,dict):
  for k,v in x.items(): assert forbidden not in str(k).lower(); scan(v)
 elif isinstance(x,list):
  for v in x: scan(v)
 elif isinstance(x,str): assert forbidden not in x.lower()
scan(m)
for k,v in {"SOURCE_OBJECT":m["source"]["uri"].split("/",3)[3],"SOURCE_SHA256":m["source"]["sha256"],"CONTRACT_OBJECT":m["cloud_contract"]["uri"].split("/",3)[3],"CONTRACT_SHA256":m["cloud_contract"]["file_sha256"],"INPUT_BUNDLE_SHA256":m["inputs"]["input_bundle_sha256"]}.items(): print(f"{k}={shlex.quote(str(v))}")
PY
)"
gcs_download "$SOURCE_OBJECT" "$WORK/source.zip"; echo "${SOURCE_SHA256}  $WORK/source.zip" | sha256sum -c -
gcs_download "$CONTRACT_OBJECT" "$WORK/fold_cloud_contract.json"; echo "${CONTRACT_SHA256}  $WORK/fold_cloud_contract.json" | sha256sum -c -
mkdir -p "$WORK/repo" "$WORK/inputs/train" "$WORK/inputs/calibration"; unzip -q "$WORK/source.zip" -d "$WORK/repo"
python3 - "$WORK/run_manifest.json" "$WORK/input_downloads.tsv" <<'PY'
import json,sys
m=json.load(open(sys.argv[1],encoding="utf-8"))
with open(sys.argv[2],"w",encoding="utf-8") as f:
 for split in ("train","calibration"):
  for e in m["inputs"][split]: f.write(f'{e["uri"].split("/",3)[3]}\t{split}/{e["index"]:03d}.jsonl\t{e["sha256"]}\n')
PY
while IFS=$'\t' read -r object relative sha; do gcs_download "$object" "$WORK/inputs/$relative"; echo "$sha  $WORK/inputs/$relative" | sha256sum -c -; done < "$WORK/input_downloads.tsv"
cd "$WORK/repo"; python3 -m venv .venv; source .venv/bin/activate; python -m pip install --upgrade pip; python -m pip install 'numpy==2.2.6' 'scikit-learn==1.8.0'
run_job(){
 local job="$1" out="$WORK/job-$(printf '%02d' "$job")" prefix="$PREFIX/results/job-$(printf '%02d' "$job")" heartbeat_pid=""
 mkdir -p "$out"
 write_status(){ python3 - "$out/status.json" "$RUN_NAME" "$job" "$1" "$MANIFEST_SHA256" "$SOURCE_SHA256" <<'PY'
import datetime,json,sys
p,run,job,state,manifest,source=sys.argv[1:]
json.dump({"schema":"hu_m43_fold_job_status_v1","run_name":run,"job_index":int(job),"state":state,"run_manifest_sha256":manifest,"source_sha256":source,"current_profile_mutated":False,"no_runtime_activation":True,"updated_at":datetime.datetime.now(datetime.timezone.utc).isoformat()},open(p,"w",encoding="utf-8"),sort_keys=True)
PY
  gcs_upload "$out/status.json" "$prefix/status.json"; }
 write_status running
 (while true; do sleep 60; python3 - "$out/heartbeat.json" "$RUN_NAME" "$job" "$MANIFEST_SHA256" "$SOURCE_SHA256" <<'PY'
import datetime,json,sys
p,run,job,manifest,source=sys.argv[1:]
json.dump({"schema":"hu_m43_fold_job_heartbeat_v1","run_name":run,"job_index":int(job),"state":"running","run_manifest_sha256":manifest,"source_sha256":source,"updated_at":datetime.datetime.now(datetime.timezone.utc).isoformat()},open(p,"w",encoding="utf-8"),sort_keys=True)
PY
  gcs_upload "$out/heartbeat.json" "$prefix/heartbeat.json" || true; done) & heartbeat_pid=$!
 local train_args=() calibration_args=() p
 for p in "$WORK"/inputs/train/*.jsonl; do train_args+=(--train "$p"); done
 for p in "$WORK"/inputs/calibration/*.jsonl; do calibration_args+=(--calibration "$p"); done
 eval "$(python3 - "$WORK/run_manifest.json" "$job" <<'PY'
import json,shlex,sys
m=json.load(open(sys.argv[1],encoding="utf-8")); j=m["jobs"][int(sys.argv[2])]
for k,v in {"JOB_SPEC_SHA256":j["job_spec_sha256"]}.items(): print(f"{k}={shlex.quote(str(v))}")
PY
)"
 set +e
 PYTHONPATH=src python -m ofc_regular.train_hu_m43_fold_job run --job-index "$job" "${train_args[@]}" "${calibration_args[@]}" --fold-cloud-contract "$WORK/fold_cloud_contract.json" --output-dir "$out" --run-name "$RUN_NAME" --run-manifest-sha256 "$MANIFEST_SHA256" --source-sha256 "$SOURCE_SHA256" --input-bundle-sha256 "$INPUT_BUNDLE_SHA256" --job-spec-sha256 "$JOB_SPEC_SHA256" >"$out/stdout.log" 2>"$out/stderr.log"
 local code=$?; set -e; kill "$heartbeat_pid" >/dev/null 2>&1 || true; wait "$heartbeat_pid" >/dev/null 2>&1 || true
 if [[ $code -ne 0 ]]; then write_status failed || true; gcs_upload "$out/stdout.log" "$prefix/stdout.log" || true; gcs_upload "$out/stderr.log" "$prefix/stderr.log" || true; return "$code"; fi
 python3 - "$WORK/run_manifest.json" "$job" "$out" "$MANIFEST_SHA256" <<'PY'
import hashlib,json,os,sys
mp,raw,out,manifest_sha=sys.argv[1:]; i=int(raw); m=json.load(open(mp,encoding="utf-8")); e=m["jobs"][i]
h=lambda p:hashlib.sha256(open(p,"rb").read()).hexdigest(); jm=json.load(open(os.path.join(out,"job_manifest.json"),encoding="utf-8")); d=json.load(open(os.path.join(out,"DONE.json"),encoding="utf-8"))
for obj,schema,status in ((jm,"hu_m43_fold_job_manifest_v1","pass"),(d,"hu_m43_fold_job_done_v1","complete")):
 assert obj["schema"]==schema and obj["status"]==status and obj["run_name"]==m["run_name"] and obj["job_index"]==i
 assert obj["job_kind"]==e["job_kind"] and obj["outer_fold"]==e["outer_fold"] and obj.get("inner_fold")==e.get("inner_fold")
 assert obj["job_spec_sha256"]==e["job_spec_sha256"] and obj["source_sha256"]==m["source"]["sha256"]
 assert obj["run_manifest_sha256"]==manifest_sha and obj["cloud_contract_sha256"]==m["cloud_contract"]["file_sha256"] and obj["input_bundle_sha256"]==m["inputs"]["input_bundle_sha256"]
 assert obj["current_profile_mutated"] is False and obj["no_runtime_activation"] is True
assert h(os.path.join(out,"estimator.pkl"))==jm["artifact_sha256"]==d["artifact_sha256"]
assert h(os.path.join(out,"job_manifest.json"))==d["job_manifest_sha256"]
PY
 gcs_upload "$out/estimator.pkl" "$prefix/estimator.pkl"; gcs_upload "$out/job_manifest.json" "$prefix/job_manifest.json"; gcs_upload "$out/stdout.log" "$prefix/stdout.log"; gcs_upload "$out/stderr.log" "$prefix/stderr.log"; write_status complete
 gcs_upload_immutable "$out/DONE.json" "$prefix/DONE.json"
}
IFS='+' read -r -a JOB_ARRAY <<< "$JOB_INDICES"; pids=(); for job in "${JOB_ARRAY[@]}"; do run_job "$job" & pids+=("$!"); done
failed=0; for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ $failed -eq 0 ]] || exit 1
echo "M4.3 fold group complete: $RUN_NAME jobs=$JOB_INDICES"
'@
        $startup = $startup -replace "`r`n", "`n"
        $startupBytes = [Text.UTF8Encoding]::new($false).GetBytes($startup)

        New-Item -ItemType Directory -Path $artifactDir -Force | Out-Null
        Write-BytesCreateNew -Path $sourcePath -Bytes ([IO.File]::ReadAllBytes($tempSource))
        Write-BytesCreateNew -Path $cloudContractPath -Bytes ([IO.File]::ReadAllBytes($preparedContract))
        Write-BytesCreateNew -Path $startupPath -Bytes $startupBytes

        $sourceSha = Get-Sha256 $sourcePath
        $jobs = @()
        $projectionJobs = @($cloudProjection.fold_plan.jobs)
        if ($projectionJobs.Count -ne $ExpectedJobs) { throw "Cloud projection does not contain exactly 30 jobs" }
        for ($i = 0; $i -lt $ExpectedJobs; $i++) {
            $projectionJob = $projectionJobs[$i]
            $slot = $i % 6
            $expectedKind = if ($slot -eq 0) { "outer_runtime" } else { "inner_oof_safety" }
            $expectedOuter = [int][Math]::Floor($i / 6)
            $expectedInner = if ($slot -eq 0) { $null } else { $slot - 1 }
            if ((Get-StrictInteger $projectionJob.job_index "projection.jobs[$i].job_index") -ne $i -or
                [string]$projectionJob.kind -ne $expectedKind -or
                (Get-StrictInteger $projectionJob.outer_fold "projection.jobs[$i].outer_fold") -ne $expectedOuter -or
                (($null -eq $expectedInner -and $null -ne $projectionJob.inner_fold) -or
                 ($null -ne $expectedInner -and (Get-StrictInteger $projectionJob.inner_fold "projection.jobs[$i].inner_fold") -ne $expectedInner))) {
                throw "Cloud projection job mapping mismatch at index $i"
            }
            $jobSpec = [ordered]@{}
            foreach ($property in $projectionJob.PSObject.Properties) {
                if ($property.Name -ne "job_spec_sha256") { $jobSpec[$property.Name] = $property.Value }
            }
            $jobs += [ordered]@{
                job_index = $i; job_kind = $expectedKind; outer_fold = $expectedOuter; inner_fold = $expectedInner
                job_spec = $jobSpec; job_spec_sha256 = [string]$projectionJob.job_spec_sha256
                shard_index = [int][Math]::Floor($i / $JobsPerShard)
                result_prefix = ("$gcsPrefix/results/job-{0:D2}" -f $i)
                done_uri = ("$gcsPrefix/results/job-{0:D2}/DONE.json" -f $i)
            }
        }
        $manifest = [ordered]@{
            schema = "hu_m43_fold_spot_run_manifest_v1"; status = "frozen"
            run_name = $RunName; project_id = $ProjectId; bucket = $Bucket; job_count = $ExpectedJobs
            source = [ordered]@{
                uri = $sourceUri; bytes = [long](Get-Item $sourcePath).Length; sha256 = $sourceSha
                fold_worker_sha256 = Get-Sha256 $foldWorkerPath; assembler_sha256 = Get-Sha256 $assemblerPath
            }
            cloud_contract = [ordered]@{
                uri = $cloudContractUri; bytes = [long](Get-Item $cloudContractPath).Length
                file_sha256 = [string]$projectionSummary.file_sha256; contract_sha256 = [string]$projectionSummary.contract_sha256
                predeclared_training_receipt_sha256 = [string]$projectionSummary.predeclared_sha256
                train_identity_sha256 = [string]$projectionSummary.train_identity_sha256
                fold_plan_sha256 = [string]$projectionSummary.fold_plan_sha256
            }
            inputs = [ordered]@{ train = $inputPlan.train; calibration = $inputPlan.calibration; input_bundle_sha256 = $inputBundleSha256 }
            jobs = $jobs
            training_config = $cloudProjection.hyperparameters
            training_config_sha256 = [string]$projectionSummary.hyperparameters_sha256
            dependencies = $cloudProjection.dependencies
            process_environment = $cloudProjection.process_environment
            source_package_policy = [ordered]@{
                mode = "ast_recursive_local_import_closure_v1"
                roots = @("src/ofc_regular/train_hu_m43_fold_job.py", "src/ofc_regular/assemble_hu_m43_fold_training.py")
                module_count = [int]$sourceClosure.Count
                entries_sha256 = [string]$sourceClosureInfo.entries_sha256
                forbidden_entry_patterns = @("(?i)l[o]cked", "(?i)\.jsonl?$", "(?i)(^|/)(outputs?|configs?|data)(/|$)")
                generic_guard_literals_non_secret = $true
                sealed_paths_content_hashes_embedded = $false
            }
            compute = [ordered]@{
                shard_count = $ExpectedShards; jobs_per_shard = $JobsPerShard; machine_type = $MachineType
                fallback_machine_types = $FallbackMachineTypes; zones = $Zones; boot_disk_gb = $BootDiskGb
                spot = $true; termination_action = "DELETE"; auto_delete = $true; vm_prefix = $vmPrefix
            }
            startup = [ordered]@{ uri = $startupUri; sha256 = Get-Sha256 $startupPath }
            cloud_input_boundary = "train_and_calibration_only"
            current_profile_mutated = $false; no_runtime_activation = $true
            created_at = (Get-Date).ToUniversalTime().ToString("o")
        }
        $manifestText = ($manifest | ConvertTo-Json -Depth 15) + "`n"
        if ($manifestText -match '(?i)locked') { throw "Cloud run manifest contains a forbidden holdout token" }
        Write-Utf8CreateNew -Path $manifestPath -Text $manifestText
        $manifestSha256 = Get-Sha256 $manifestPath
        New-LocalSealedRebind -TrainPaths $resolvedTrain -CalibrationPaths $resolvedCalibration `
            -DataContractPath $resolvedContract -PlanPath $resolvedPlan -RepoRoot $repoRoot `
            -PredeclaredReceiptPath $predeclaredReceipt -CloudContractPath $cloudContractPath `
            -RunManifestPath $manifestPath -SourceArchivePath $sourcePath -OutputPath $localRebindPath
        if ($DryRun) {
            $boundaryAuditScript = @'
import json,pathlib,re,sys
contract,plan,package,projection,startup,manifest=sys.argv[1:]
def collect(node,inside=False,key=""):
    values=[]; now=inside or ("locked" in key.lower())
    if isinstance(node,dict):
        for k,v in node.items(): values.extend(collect(v,now,str(k)))
    elif isinstance(node,list):
        for v in node: values.extend(collect(v,now,key))
    elif now and isinstance(node,str):
        lowered=key.lower()
        if any(marker in lowered for marker in ("path","sha","hash","identity")) or re.fullmatch(r"[0-9a-fA-F]{64}",node) or ".jsonl" in node.lower() or "\\" in node or "/" in node:
            values.append(node)
    return values
values=[]
for path in (contract,plan): values.extend(collect(json.load(open(path,encoding="utf-8-sig"))))
values=sorted({value for value in values if value})
targets=[]
for path in pathlib.Path(package).rglob("*.py"): targets.append((str(path.relative_to(package)).replace("\\","/"),path.read_text(encoding="utf-8")))
for name,path in (("fold_cloud_contract.json",projection),("startup_hu_m43_fold_group.sh",startup),("m43_fold_run_manifest.json",manifest)):
    targets.append((name,pathlib.Path(path).read_text(encoding="utf-8")))
matches=[]
for name,text in targets:
    folded=text.casefold()
    for value in values:
        if value.casefold() in folded: matches.append({"artifact":name,"value_length":len(value)})
if matches: raise ValueError(f"sealed locked-subtree value leaked into cloud-safe artifact: {matches}")
print(json.dumps({"schema":"hu_m43_dryrun_cloud_boundary_audit_v1","status":"pass","sensitive_value_count":len(values),"scanned_source_modules":sum(1 for n,_ in targets if n.endswith('.py')),"scanned_cloud_safe_artifacts":["source_closure","fold_cloud_contract.json","startup_hu_m43_fold_group.sh","m43_fold_run_manifest.json"],"exact_sensitive_value_matches":0,"generic_guard_literals_non_secret":True,"sealed_paths_content_hashes_embedded":False},sort_keys=True))
'@
            $oldPreference = $ErrorActionPreference; $ErrorActionPreference = "Continue"
            try { $auditOutput = @($boundaryAuditScript | & python - $resolvedContract $resolvedPlan $packageRoot $cloudContractPath $startupPath $manifestPath 2>&1); $auditCode = $LASTEXITCODE }
            finally { $ErrorActionPreference = $oldPreference }
            if ($auditCode -ne 0) { throw "DryRun cloud boundary audit failed: $($auditOutput -join [Environment]::NewLine)" }
            $dryRunBoundaryAudit = (@($auditOutput) -join "`n") | ConvertFrom-Json
        }
    }
    finally { Remove-Item -LiteralPath $temp -Recurse -Force -ErrorAction SilentlyContinue }
}

Assert-FrozenManifest $manifest
Assert-SourcePackagePolicy -Manifest $manifest -ArchivePath $sourcePath
$plan = [ordered]@{
    schema = "hu_m43_fold_spot_start_plan_v1"; run_name = $RunName; manifest_uri = $manifestUri
    manifest_sha256 = $manifestSha256; job_count = $ExpectedJobs; shard_count = $ExpectedShards
    selected_shards = $uniqueShards; create_instances = [bool]$CreateInstances; resume_existing = [bool]$ResumeExisting
    dry_run = [bool]$DryRun; artifact_directory = $artifactDir
    cloud_safe_artifacts = @("ofc_regular_hu_m43_fold_source.zip", "fold_cloud_contract.json", "startup_hu_m43_fold_group.sh", "m43_fold_run_manifest.json")
    local_only_artifact = "m43_local_sealed_rebind_manifest.json"; local_only_upload_forbidden = $true
    current_profile_mutated = $false; no_runtime_activation = $true
}
if ($DryRun) {
    $plan.boundary_audit = $dryRunBoundaryAudit
    Write-Utf8CreateNew -Path $dryRunPlanPath -Text (($plan | ConvertTo-Json -Depth 12) + "`n")
    $plan | ConvertTo-Json -Depth 12
    exit 0
}

if (-not $ResumeExisting) {
    $localInputPaths = [ordered]@{ train = $resolvedTrain; calibration = $resolvedCalibration }
    foreach ($split in @("train", "calibration")) {
        $splitPaths = @($localInputPaths[$split])
        for ($i = 0; $i -lt $splitPaths.Count; $i++) {
            Invoke-Gcloud @("storage", "cp", $splitPaths[$i], $manifest.inputs.$split[$i].uri, "--project", $ProjectId, "--if-generation-match=0") | Out-Null
        }
    }
    foreach ($pair in @(
        @($sourcePath, $sourceUri), @($cloudContractPath, $cloudContractUri),
        @($startupPath, $startupUri), @($manifestPath, $manifestUri)
    )) { Invoke-Gcloud @("storage", "cp", $pair[0], $pair[1], "--project", $ProjectId, "--if-generation-match=0") | Out-Null }
}
if (-not $CreateInstances) { $plan.uploaded = $true; $plan | ConvertTo-Json -Depth 12; exit 0 }

$attempts = [Collections.Generic.List[object]]::new()
$createdShards = @()
foreach ($shard in $uniqueShards) {
    $start = $shard * $JobsPerShard
    $end = [Math]::Min($ExpectedJobs - 1, $start + $JobsPerShard - 1)
    $missing = @()
    for ($job = $start; $job -le $end; $job++) {
        if (Test-GcsObject $manifest.jobs[$job].done_uri) {
            $existingDone = Get-GcsJson $manifest.jobs[$job].done_uri
            Assert-FrozenDone $existingDone $manifest.jobs[$job] $job $manifest $manifestSha256
        }
        else { $missing += $job }
    }
    if ($missing.Count -eq 0) { continue }
    $vmName = ("{0}-g{1:D2}" -f $vmPrefix, $shard)
    if (@(Get-InstancesByName $vmName).Count -ne 0) { throw "Worker already exists for shard ${shard}: $vmName" }
    $metadata = @(
        "BUCKET=$Bucket", "RUN_NAME=$RunName", "MANIFEST_OBJECT=runs/$RunName/source/m43_fold_run_manifest.json",
        "MANIFEST_SHA256=$manifestSha256", ("JOB_INDICES=" + ($missing -join "+"))
    ) -join ","
    $created = $false
    foreach ($machine in @($MachineType) + @($FallbackMachineTypes)) {
        $diskType = if ($machine -like "c4-*") { "hyperdisk-balanced" } else { "pd-balanced" }
        foreach ($zone in $Zones) {
            $args = @(
                "compute", "instances", "create", $vmName, "--project", $ProjectId, "--zone", $zone,
                "--machine-type", $machine, "--provisioning-model", "SPOT", "--instance-termination-action", "DELETE",
                "--maintenance-policy", "TERMINATE", "--boot-disk-size", ("{0}GB" -f $BootDiskGb),
                "--boot-disk-type", $diskType, "--image-family", "ubuntu-2404-lts-amd64", "--image-project", "ubuntu-os-cloud",
                "--scopes", "cloud-platform", "--metadata", $metadata,
                "--metadata-from-file", ("startup-script={0}" -f $startupPath), "--quiet"
            )
            $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
            try { $output = & gcloud @args 2>&1; $code = $LASTEXITCODE }
            finally { $ErrorActionPreference = $old }
            $attempts.Add([ordered]@{ shard_index = $shard; jobs = $missing; machine_type = $machine; zone = $zone; exit_code = $code; output = @($output) })
            if ($code -eq 0) { $created = $true; $createdShards += $shard; break }
            if (@(Get-InstancesByName $vmName).Count -ne 0) { throw "Ambiguous create failure left $vmName present" }
        }
        if ($created) { break }
    }
    if (-not $created) { throw "Unable to create Spot worker for M4.3 shard $shard" }
}
$plan.uploaded = $true; $plan.created_shards = $createdShards; $plan.attempts = $attempts
$plan | ConvertTo-Json -Depth 15
