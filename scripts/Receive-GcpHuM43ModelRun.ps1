param(
    [Parameter(Mandatory = $true)][string]$RunName,
    [Parameter(Mandatory = $true)][string[]]$Train,
    [Parameter(Mandatory = $true)][string[]]$Calibration,
    [Parameter(Mandatory = $true)][string]$M43DataContract,
    [string]$M43Plan = "configs/hu_joint_policy_m43_pilot.json",
    [string]$ProjectId = "ofc-solver-485418",
    [string]$Bucket = "pokerhu-ofc-solver-485418-training",
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"
$ExpectedJobs = 30
$ExpectedPredeclaredReceiptSha256 = "9313bdc6ffa33335032ee3d98e279d123d2f3aa70571114e75ee5f38e2594bea"

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-TextSha256 {
    param([Parameter(Mandatory = $true)][string]$Text)
    $sha = [Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($sha.ComputeHash([Text.UTF8Encoding]::new($false).GetBytes($Text)))).Replace("-", "").ToLowerInvariant() }
    finally { $sha.Dispose() }
}

function Resolve-InputPath {
    param([Parameter(Mandatory = $true)][string]$Path, [Parameter(Mandatory = $true)][string]$RepoRoot)
    $candidate = if ([IO.Path]::IsPathRooted($Path)) { $Path } else { Join-Path $RepoRoot $Path }
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) { throw "Required local assembly input is missing: $candidate" }
    return (Resolve-Path -LiteralPath $candidate).Path
}

function Invoke-Gcloud {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    try { $output = & gcloud @Arguments 2>&1; $code = $LASTEXITCODE }
    finally { $ErrorActionPreference = $old }
    if ($code -ne 0) { throw "gcloud failed ($code): gcloud $($Arguments -join ' ')`n$(@($output) -join [Environment]::NewLine)" }
    return @($output)
}

function Get-GcsUris {
    param([Parameter(Mandatory = $true)][string]$Pattern)
    $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    try { $raw = & gcloud storage ls $Pattern 2>&1; $code = $LASTEXITCODE }
    finally { $ErrorActionPreference = $old }
    if ($code -ne 0) { throw "All 30 immutable DONE objects are required before receive: $(@($raw) -join [Environment]::NewLine)" }
    return @($raw | ForEach-Object { ([string]$_).Trim() } | Where-Object { $_ })
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
    if ($Value -is [Collections.IDictionary]) { return @($Value.Keys | ForEach-Object { [string]$_ }) }
    return @($Value.PSObject.Properties.Name)
}

function Assert-SourcePackagePolicy {
    param([Parameter(Mandatory = $true)]$Manifest, [Parameter(Mandatory = $true)][string]$ArchivePath)
    $policy = $Manifest.source_package_policy
    $roots = @("src/ofc_regular/train_hu_m43_fold_job.py", "src/ofc_regular/assemble_hu_m43_fold_training.py")
    $patterns = @("(?i)l[o]cked", "(?i)\.jsonl?$", "(?i)(^|/)(outputs?|configs?|data)(/|$)")
    if ([string]$policy.mode -ne "ast_recursive_local_import_closure_v1" -or (@($policy.roots) -join "|") -ne ($roots -join "|") -or
        (@($policy.forbidden_entry_patterns) -join "|") -ne ($patterns -join "|") -or
        $policy.generic_guard_literals_non_secret -ne $true -or $policy.sealed_paths_content_hashes_embedded -ne $false -or
        (Get-StrictInteger $policy.module_count "source_package_policy.module_count") -lt 3 -or [string]$policy.entries_sha256 -notmatch '^[0-9a-f]{64}$') { throw "Frozen source package policy changed" }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [IO.Compression.ZipFile]::OpenRead($ArchivePath)
    try {
        if (@($archive.Entries | Where-Object { $_.FullName -match '\\' }).Count -ne 0) { throw "Frozen source archive has non-portable entry separators" }
        [string[]]$entries = @($archive.Entries | Where-Object { $_.FullName -match '\.py$' } | ForEach-Object { $_.FullName })
        [Array]::Sort($entries, [StringComparer]::Ordinal)
        if ($entries.Count -ne (Get-StrictInteger $policy.module_count "source_package_policy.module_count") -or
            @($entries | Where-Object { $_ -match '(?i)l[o]cked|\.jsonl?$|(^|/)(outputs?|configs?|data)(/|$)' }).Count -ne 0 -or
            (Get-TextSha256 ('["' + ($entries -join '","') + '"]')) -ne [string]$policy.entries_sha256) { throw "Frozen source archive violates package policy" }
    }
    finally { $archive.Dispose() }
}

function Assert-Manifest {
    param([Parameter(Mandatory = $true)]$Manifest)
    if ($Manifest.schema -ne "hu_m43_fold_spot_run_manifest_v1" -or $Manifest.status -ne "frozen" -or
        $Manifest.run_name -ne $RunName -or $Manifest.project_id -ne $ProjectId -or $Manifest.bucket -ne $Bucket -or
        (Get-StrictInteger $Manifest.job_count "job_count") -ne $ExpectedJobs -or $Manifest.current_profile_mutated -ne $false -or
        $Manifest.no_runtime_activation -ne $true -or (Get-StrictInteger $Manifest.compute.shard_count "compute.shard_count") -ne 8 -or
        (Get-StrictInteger $Manifest.compute.jobs_per_shard "compute.jobs_per_shard") -ne 4 -or $Manifest.compute.auto_delete -ne $true) {
        throw "Frozen M4.3 fold manifest identity/schema is invalid"
    }
    $cloudKeys = @(Get-ObjectPropertyNames $Manifest.cloud_contract | Sort-Object)
    $expectedCloudKeys = @(@("bytes", "contract_sha256", "file_sha256", "fold_plan_sha256", "predeclared_training_receipt_sha256", "train_identity_sha256", "uri") | Sort-Object)
    if (($cloudKeys -join "|") -ne ($expectedCloudKeys -join "|") -or
        [string]$Manifest.source.uri -ne "gs://$Bucket/runs/$RunName/source/ofc_regular_hu_m43_fold_source.zip" -or
        (Get-StrictInteger $Manifest.source.bytes "source.bytes") -lt 1 -or [string]$Manifest.source.sha256 -notmatch '^[0-9a-f]{64}$' -or
        [string]$Manifest.cloud_contract.uri -ne "gs://$Bucket/runs/$RunName/inputs/fold_cloud_contract.json" -or
        (Get-StrictInteger $Manifest.cloud_contract.bytes "cloud_contract.bytes") -lt 1 -or
        [string]$Manifest.dependencies.numpy -ne "2.2.6" -or [string]$Manifest.dependencies.scikit_learn -ne "1.8.0") {
        throw "Frozen source/cloud/dependency binding is invalid"
    }
    foreach ($name in @("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")) {
        if ([string]$Manifest.process_environment.$name -ne "1") { throw "Frozen deterministic environment changed: $name" }
    }
    if ([string]$Manifest.process_environment.PYTHONHASHSEED -ne "0") { throw "Frozen PYTHONHASHSEED changed" }
    $sourcePolicy = $Manifest.source_package_policy
    $roots = @("src/ofc_regular/train_hu_m43_fold_job.py", "src/ofc_regular/assemble_hu_m43_fold_training.py")
    $patterns = @("(?i)l[o]cked", "(?i)\.jsonl?$", "(?i)(^|/)(outputs?|configs?|data)(/|$)")
    if ([string]$sourcePolicy.mode -ne "ast_recursive_local_import_closure_v1" -or (@($sourcePolicy.roots) -join "|") -ne ($roots -join "|") -or
        (@($sourcePolicy.forbidden_entry_patterns) -join "|") -ne ($patterns -join "|") -or
        (Get-StrictInteger $sourcePolicy.module_count "source_package_policy.module_count") -lt 3 -or [string]$sourcePolicy.entries_sha256 -notmatch '^[0-9a-f]{64}$' -or
        $sourcePolicy.generic_guard_literals_non_secret -ne $true -or $sourcePolicy.sealed_paths_content_hashes_embedded -ne $false) { throw "Frozen source package policy is invalid" }
    $jobs = @($Manifest.jobs)
    if ($jobs.Count -ne $ExpectedJobs) { throw "M4.3 manifest must contain exactly 30 jobs" }
    $seen = [Collections.Generic.HashSet[int]]::new()
    for ($i = 0; $i -lt $ExpectedJobs; $i++) {
        $job = $jobs[$i]; $slot = $i % 6
        $kind = if ($slot -eq 0) { "outer_runtime" } else { "inner_oof_safety" }
        $outer = [Math]::Floor($i / 6)
        $inner = if ($slot -eq 0) { $null } else { $slot - 1 }
        $jobIndex = Get-StrictInteger $job.job_index "jobs[$i].job_index"
        $outerFold = Get-StrictInteger $job.outer_fold "jobs[$i].outer_fold"
        $shardIndex = Get-StrictInteger $job.shard_index "jobs[$i].shard_index"
        $innerFold = if ($null -eq $job.inner_fold) { $null } else { Get-StrictInteger $job.inner_fold "jobs[$i].inner_fold" }
        $jobSpec = $job.job_spec
        $specInner = if ($null -eq $jobSpec.inner_fold) { $null } else { Get-StrictInteger $jobSpec.inner_fold "jobs[$i].job_spec.inner_fold" }
        foreach ($field in @("job_index", "outer_fold", "estimator_fold_index", "estimator_seed", "fit_samples", "outer_validation_samples", "inner_validation_samples")) {
            Get-StrictInteger $jobSpec.$field "jobs[$i].job_spec.$field" | Out-Null
        }
        if (-not $seen.Add([int]$jobIndex) -or $jobIndex -ne $i -or
            [string]$job.job_kind -ne $kind -or $outerFold -ne $outer -or $shardIndex -ne [Math]::Floor($i / 4) -or
            (($null -eq $inner -and $null -ne $job.inner_fold) -or ($null -ne $inner -and $innerFold -ne $inner)) -or
            (Get-StrictInteger $jobSpec.job_index "job_spec.job_index") -ne $i -or [string]$jobSpec.kind -ne $kind -or
            (Get-StrictInteger $jobSpec.outer_fold "job_spec.outer_fold") -ne $outer -or
            (($null -eq $inner -and $null -ne $jobSpec.inner_fold) -or ($null -ne $inner -and $specInner -ne $inner)) -or
            [string]$job.done_uri -ne ("gs://$Bucket/runs/$RunName/results/job-{0:D2}/DONE.json" -f $i)) {
            throw "Frozen job mapping is invalid at index $i"
        }
    }
    foreach ($split in @("train", "calibration")) {
        $entries = @($Manifest.inputs.$split)
        for ($i = 0; $i -lt $entries.Count; $i++) {
            if ((Get-StrictInteger $entries[$i].index "inputs.$split[$i].index") -ne $i -or
                (Get-StrictInteger $entries[$i].bytes "inputs.$split[$i].bytes") -lt 1 -or
                (Get-StrictInteger $entries[$i].rows "inputs.$split[$i].rows") -lt 1) { throw "Invalid input integer provenance" }
        }
    }
    $hp = $Manifest.training_config
    if ((Get-StrictInteger $hp.cross_fit_folds "cross_fit_folds") -ne 5 -or
        (Get-StrictInteger $hp.iterations "iterations") -ne 150 -or
        (Get-StrictInteger $hp.max_leaf_nodes "max_leaf_nodes") -ne 31 -or
        (Get-StrictInteger $hp.seed "seed") -ne 2026071801 -or
        (Get-StrictInteger $hp.safety_split_seed "safety_split_seed") -ne 2026071802 -or
        (Get-StrictInteger $hp.minimum_safety_fit_samples "minimum_safety_fit_samples") -ne 30 -or
        (Get-StrictInteger $hp.minimum_threshold_lock_samples "minimum_threshold_lock_samples") -ne 30 -or
        (Get-StrictInteger $hp.minimum_calibration_fires "minimum_calibration_fires") -ne 10 -or
        [string]$hp.action_score_mode -ne "baseline_paired_delta_risk_ensemble_v3" -or
        [string]$hp.model_id -ne "hu-m43-t1-second-regular-hu-m43-c2e16-pilot200-20260713-1810" -or
        [double]$hp.positive_gain_score_weight -ne 0.25 -or [double]$hp.downside_risk_score_weight -ne 0.5 -or
        [double]$hp.ensemble_disagreement_score_weight -ne 0.25) {
        throw "Frozen full M4.3 training configuration changed"
    }
}

if ($RunName -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$') { throw "RunName is not path-safe" }
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runDir = Join-Path $repoRoot "outputs/gcp_runs/$RunName"
$manifestPath = Join-Path $runDir "m43_fold_run_manifest.json"
$cloudContractPath = Join-Path $runDir "fold_cloud_contract.json"
$localRebindPath = Join-Path $runDir "m43_local_sealed_rebind_manifest.json"
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf) -or
    -not (Test-Path -LiteralPath $cloudContractPath -PathType Leaf) -or
    -not (Test-Path -LiteralPath $localRebindPath -PathType Leaf)) {
    throw "Local frozen manifest/cloud contract/local-only sealed rebind is missing for $RunName"
}
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$manifestSha = Get-Sha256 $manifestPath
Assert-Manifest $manifest
$projectionRaw = Get-Content -LiteralPath $cloudContractPath -Raw
if ($projectionRaw -match '(?i)locked') { throw "Cloud contract contains a forbidden holdout token" }
$projection = $projectionRaw | ConvertFrom-Json
if ($projection.schema -ne "hu_m43_fold_cloud_contract_v1" -or $projection.status -ne "frozen_cloud_safe" -or
    $projection.cloud_safe -ne $true -or (Get-Sha256 $cloudContractPath) -ne [string]$manifest.cloud_contract.file_sha256 -or
    [string]$projection.contract_sha256 -ne [string]$manifest.cloud_contract.contract_sha256) {
    throw "Local cloud-safe projection is stale or invalid"
}

$resolvedTrain = @($Train | ForEach-Object { Resolve-InputPath $_ $repoRoot })
$resolvedCalibration = @($Calibration | ForEach-Object { Resolve-InputPath $_ $repoRoot })
$resolvedContract = Resolve-InputPath $M43DataContract $repoRoot
$resolvedPlan = Resolve-InputPath $M43Plan $repoRoot
$predeclaredReceipt = Resolve-InputPath "outputs/hu_joint_policy/m43_spot/regular-hu-m43-c2e16-pilot200-20260713-1810/m43_t1_second_attempt01/local_training_aborted.json" $repoRoot
if ((Get-Sha256 $predeclaredReceipt) -ne $ExpectedPredeclaredReceiptSha256 -or
    [string]$projection.predeclared_training_receipt_sha256 -ne $ExpectedPredeclaredReceiptSha256 -or
    [string]$manifest.cloud_contract.predeclared_training_receipt_sha256 -ne $ExpectedPredeclaredReceiptSha256) {
    throw "Predeclared training receipt provenance changed"
}
$configBinding = @'
import hashlib,json,sys
m=json.load(open(sys.argv[1],encoding="utf-8")); p=json.load(open(sys.argv[2],encoding="utf-8"))
c=lambda value:hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()
unsigned=dict(p); declared=unsigned.pop("contract_sha256"); assert declared==c(unsigned)
assert m["training_config"]==p["hyperparameters"] and m["training_config_sha256"]==c(p["hyperparameters"])
assert m["dependencies"]==p["dependencies"]=={"numpy":"2.2.6","scikit_learn":"1.8.0"}
assert m["process_environment"]==p["process_environment"]
assert m["inputs"]["input_bundle_sha256"]==p["input_bundle_sha256"]
for split in ("train","calibration"):
 assert [{k:v for k,v in e.items() if k!="uri"} for e in m["inputs"][split]]==p["inputs"][split]
assert len(m["jobs"])==len(p["fold_plan"]["jobs"])==30
for i,(actual,projected) in enumerate(zip(m["jobs"],p["fold_plan"]["jobs"])):
 assert actual["job_spec"]=={k:v for k,v in projected.items() if k!="job_spec_sha256"} and actual["job_spec_sha256"]==projected["job_spec_sha256"]
 assert type(actual["job_index"]) is int and actual["job_index"]==i
'@
$oldPreference = $ErrorActionPreference; $ErrorActionPreference = "Continue"
try { $configOutput = @($configBinding | & python - $manifestPath $cloudContractPath 2>&1); $configCode = $LASTEXITCODE }
finally { $ErrorActionPreference = $oldPreference }
if ($configCode -ne 0) { throw "Run manifest and cloud projection config differ: $($configOutput -join [Environment]::NewLine)" }
foreach ($pair in @(@("train", $resolvedTrain), @("calibration", $resolvedCalibration))) {
    $split = $pair[0]; $paths = @($pair[1]); $expected = @($manifest.inputs.$split)
    if ($paths.Count -ne $expected.Count) { throw "Local $split input count differs from the frozen run" }
    for ($i = 0; $i -lt $paths.Count; $i++) {
        if ((Get-Sha256 $paths[$i]) -ne [string]$expected[$i].sha256 -or
            [long](Get-Item -LiteralPath $paths[$i]).Length -ne [long]$expected[$i].bytes -or [int]$expected[$i].index -ne $i) {
            throw "Local $split input provenance mismatch at index $i"
        }
    }
}

$modelRunsRoot = [IO.Path]::GetFullPath((Join-Path $repoRoot "outputs/hu_joint_policy/m43_fold_model_runs"))
New-Item -ItemType Directory -Path $modelRunsRoot -Force | Out-Null
$stageDir = Join-Path $modelRunsRoot (".receiving-{0}-{1}" -f $RunName, [guid]::NewGuid().ToString("N"))
$foldArtifactsDir = Join-Path $stageDir "fold_artifacts"
New-Item -ItemType Directory -Path $foldArtifactsDir -Force | Out-Null
$frozenSourcePath = Join-Path $stageDir "ofc_regular_hu_m43_fold_source.zip"
$localFrozenSourcePath = Join-Path $runDir "ofc_regular_hu_m43_fold_source.zip"
if (Test-Path -LiteralPath $localFrozenSourcePath -PathType Leaf) {
    if ((Get-Sha256 $localFrozenSourcePath) -ne [string]$manifest.source.sha256) { throw "Local frozen source archive hash mismatch" }
    Copy-Item -LiteralPath $localFrozenSourcePath -Destination $frozenSourcePath
}
else { Invoke-Gcloud @("storage", "cp", "$($manifest.source.uri)", $frozenSourcePath, "--project", $ProjectId) | Out-Null }
if ((Get-Sha256 $frozenSourcePath) -ne [string]$manifest.source.sha256 -or
    [long](Get-Item -LiteralPath $frozenSourcePath).Length -ne (Get-StrictInteger $manifest.source.bytes "source.bytes")) {
    throw "Downloaded frozen source archive hash/size mismatch"
}
Assert-SourcePackagePolicy -Manifest $manifest -ArchivePath $frozenSourcePath
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [IO.Compression.ZipFile]::OpenRead($frozenSourcePath)
try {
    $unsafeEntries = @($zip.Entries | Where-Object {
        $_.FullName -and $_.FullName -notmatch '^src/?$|^src/ofc_regular/?$|^src/ofc_regular/[^/\\]+\.py$'
    })
    if ($unsafeEntries.Count -ne 0) { throw "Frozen source archive contains an unexpected or unsafe entry" }
}
finally { $zip.Dispose() }
$frozenRepo = Join-Path $stageDir "frozen_repo"
Expand-Archive -LiteralPath $frozenSourcePath -DestinationPath $frozenRepo
$frozenFoldWorker = Join-Path $frozenRepo "src/ofc_regular/train_hu_m43_fold_job.py"
$frozenAssembler = Join-Path $frozenRepo "src/ofc_regular/assemble_hu_m43_fold_training.py"
if (-not (Test-Path $frozenFoldWorker -PathType Leaf) -or -not (Test-Path $frozenAssembler -PathType Leaf) -or
    (Get-Sha256 $frozenFoldWorker) -ne [string]$manifest.source.fold_worker_sha256 -or
    (Get-Sha256 $frozenAssembler) -ne [string]$manifest.source.assembler_sha256) {
    throw "Frozen source archive module hash chain is invalid"
}
$frozenPythonPath = Join-Path $frozenRepo "src"
$hp = $manifest.training_config
$thresholdArgument = (@($hp.thresholds | ForEach-Object { [string]::Format([Globalization.CultureInfo]::InvariantCulture, "{0:G17}", [double]$_) }) -join ",")
$rebindArgs = @("-m", "ofc_regular.train_hu_m43_fold_job", "rebind", "--run-name", $RunName)
foreach ($path in $resolvedTrain) { $rebindArgs += @("--train", $path) }
foreach ($path in $resolvedCalibration) { $rebindArgs += @("--calibration", $path) }
$rebindArgs += @(
    "--m43-data-contract", $resolvedContract, "--m43-plan", $resolvedPlan, "--repo-root", $repoRoot,
    "--predeclared-receipt", $predeclaredReceipt, "--fold-cloud-contract", $cloudContractPath,
    "--run-manifest", $manifestPath, "--source-archive", $frozenSourcePath,
    "--output-rebind-manifest", (Join-Path $stageDir "candidate_local_sealed_rebind_manifest.json"),
    "--action-score-mode", "$($hp.action_score_mode)", "--model-id", "$($hp.model_id)",
    "--cross-fit-folds", "$($hp.cross_fit_folds)", "--iterations", "$($hp.iterations)",
    "--max-leaf-nodes", "$($hp.max_leaf_nodes)", "--learning-rate", "$($hp.learning_rate)",
    "--l2-regularization", "$($hp.l2_regularization)", "--seed", "$($hp.seed)",
    "--paired-se-floor", "$($hp.paired_se_floor)", "--paired-huber-alpha", "$($hp.paired_huber_alpha)",
    "--downside-quantile", "$($hp.downside_quantile)", "--positive-gain-score-weight", "$($hp.positive_gain_score_weight)",
    "--downside-risk-score-weight", "$($hp.downside_risk_score_weight)",
    "--ensemble-disagreement-score-weight", "$($hp.ensemble_disagreement_score_weight)",
    "--near-best-margin", "$($hp.near_best_margin)", "--minimum-safe-teacher-gain", "$($hp.minimum_safe_teacher_gain)",
    "--safety-calibrator-c", "$($hp.safety_calibrator_c)", "--safety-fit-ratio", "$($hp.safety_fit_ratio)",
    "--safety-split-seed", "$($hp.safety_split_seed)", "--minimum-safety-fit-samples", "$($hp.minimum_safety_fit_samples)",
    "--minimum-threshold-lock-samples", "$($hp.minimum_threshold_lock_samples)", "--thresholds", $thresholdArgument,
    "--minimum-calibration-fires", "$($hp.minimum_calibration_fires)",
    "--maximum-false-positive-rate", "$($hp.maximum_false_positive_rate)", "--maximum-p95-loss", "$($hp.maximum_p95_loss)",
    "--maximum-p99-loss", "$($hp.maximum_p99_loss)", "--maximum-max-loss", "$($hp.maximum_max_loss)"
)
$candidateRebindPath = Join-Path $stageDir "candidate_local_sealed_rebind_manifest.json"
$rebindEnvironment = [ordered]@{ PYTHONPATH = $frozenPythonPath; PYTHONHASHSEED = "0"; OMP_NUM_THREADS = "1"; OPENBLAS_NUM_THREADS = "1"; MKL_NUM_THREADS = "1"; NUMEXPR_NUM_THREADS = "1" }
$oldRebindEnvironment = @{}
foreach ($name in $rebindEnvironment.Keys) {
    $oldRebindEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
    [Environment]::SetEnvironmentVariable($name, [string]$rebindEnvironment[$name], "Process")
}
$oldPreference = $ErrorActionPreference; $ErrorActionPreference = "Continue"
$rebindImportPreflight = @'
import pathlib,sys
root=pathlib.Path(sys.argv[1]).resolve()
import ofc_regular,ofc_regular.train_hu_m43_fold_job as worker,ofc_regular.assemble_hu_m43_fold_training as assembler
import numpy,sklearn
for module in (ofc_regular,worker,assembler): pathlib.Path(module.__file__).resolve().relative_to(root)
assert numpy.__version__=="2.2.6" and sklearn.__version__=="1.8.0"
'@
try {
    $rebindImportOutput = @($rebindImportPreflight | & python - $frozenPythonPath 2>&1)
    if ($LASTEXITCODE -ne 0) { throw "Frozen source import/dependency preflight failed before receive: $($rebindImportOutput -join [Environment]::NewLine)" }
    $rebindOutput = @(& python @rebindArgs 2>&1); $rebindCode = $LASTEXITCODE
}
finally {
    foreach ($name in $oldRebindEnvironment.Keys) { [Environment]::SetEnvironmentVariable($name, $oldRebindEnvironment[$name], "Process") }
    $ErrorActionPreference = $oldPreference
}
if ($rebindCode -ne 0) { throw "Frozen source local-only rebind verification failed: $($rebindOutput -join [Environment]::NewLine)" }
if (-not (Test-Path -LiteralPath $candidateRebindPath -PathType Leaf) -or
    [long](Get-Item -LiteralPath $candidateRebindPath).Length -ne [long](Get-Item -LiteralPath $localRebindPath).Length -or
    (Get-Sha256 $candidateRebindPath) -ne (Get-Sha256 $localRebindPath)) {
    throw "Original local-only sealed rebind differs from the frozen-source reconstruction"
}
Remove-Item -LiteralPath $candidateRebindPath -Force

if ($OutputDir) {
    if (-not [IO.Path]::IsPathRooted($OutputDir)) { $OutputDir = Join-Path $repoRoot $OutputDir }
    $OutputDir = [IO.Path]::GetFullPath($OutputDir)
}
else { $OutputDir = [IO.Path]::GetFullPath((Join-Path $modelRunsRoot $RunName)) }
$rootPrefix = $modelRunsRoot.TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
if (-not $OutputDir.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "OutputDir must remain under m43_fold_model_runs; refusing a profile/current overwrite"
}
if (Test-Path -LiteralPath $OutputDir) {
    $receiptPath = Join-Path $OutputDir "receipt.json"
    $modelPath = Join-Path $OutputDir "model.pkl"
    $trainingManifestPath = Join-Path $OutputDir "training_manifest.json"
    $outputRebindPath = Join-Path $OutputDir "m43_local_sealed_rebind_manifest.json"
    if (-not (Test-Path $receiptPath -PathType Leaf) -or -not (Test-Path $modelPath -PathType Leaf) -or
        -not (Test-Path $trainingManifestPath -PathType Leaf) -or -not (Test-Path $outputRebindPath -PathType Leaf)) {
        throw "OutputDir exists without a complete verified receipt; refusing overwrite"
    }
    $existing = Get-Content $receiptPath -Raw | ConvertFrom-Json
    if ($existing.schema -ne "hu_m43_fold_model_receipt_v1" -or $existing.status -ne "verified" -or
        $existing.run_name -ne $RunName -or $existing.run_manifest_sha256 -ne $manifestSha -or
        (Get-Sha256 $modelPath) -ne $existing.model_sha256 -or
        (Get-Sha256 $trainingManifestPath) -ne $existing.training_manifest_sha256 -or
        (Get-Sha256 $outputRebindPath) -ne $existing.local_sealed_rebind_manifest_file_sha256 -or
        (Get-Sha256 $localRebindPath) -ne $existing.local_sealed_rebind_manifest_file_sha256 -or
        $existing.current_profile_mutated -ne $false -or $existing.no_runtime_activation -ne $true) {
        throw "Existing receipt is stale or its hash chain is broken"
    }
    Remove-Item -LiteralPath $stageDir -Recurse -Force
    $existing | ConvertTo-Json -Depth 15
    exit 0
}

$prefix = "gs://$Bucket/runs/$RunName/results"
$donePattern = "^" + [regex]::Escape($prefix) + "/job-(\d{2})/DONE\.json$"
$doneUris = @(Get-GcsUris "$prefix/**/DONE.json")
if ($doneUris.Count -ne $ExpectedJobs) { throw "Receive requires exactly 30 DONE URIs; found $($doneUris.Count)" }
$seenDoneJobs = [Collections.Generic.HashSet[int]]::new()
foreach ($uri in $doneUris) {
    $match = [regex]::Match($uri, $donePattern)
    if (-not $match.Success) { throw "Unexpected DONE URI: $uri" }
    $jobIndex = [int]$match.Groups[1].Value
    if ($jobIndex -lt 0 -or $jobIndex -ge $ExpectedJobs -or -not $seenDoneJobs.Add($jobIndex)) { throw "Out-of-range or duplicate DONE identity: $uri" }
    if ($uri -ne [string]$manifest.jobs[$jobIndex].done_uri) { throw "DONE URI does not match frozen job mapping: $uri" }
}

$verifiedJobs = @()
for ($i = 0; $i -lt $ExpectedJobs; $i++) {
    $jobDir = Join-Path $foldArtifactsDir ("job-{0:D2}" -f $i)
    New-Item -ItemType Directory -Path $jobDir | Out-Null
    $jobPrefix = "$prefix/job-{0:D2}" -f $i
    foreach ($name in @("estimator.pkl", "job_manifest.json", "DONE.json")) {
        Invoke-Gcloud @("storage", "cp", "$jobPrefix/$name", (Join-Path $jobDir $name), "--project", $ProjectId) | Out-Null
    }
    $estimatorPath = Join-Path $jobDir "estimator.pkl"
    $jobManifestPath = Join-Path $jobDir "job_manifest.json"
    $donePath = Join-Path $jobDir "DONE.json"
    $jobManifest = Get-Content -LiteralPath $jobManifestPath -Raw | ConvertFrom-Json
    $done = Get-Content -LiteralPath $donePath -Raw | ConvertFrom-Json
    $expected = $manifest.jobs[$i]
    foreach ($check in @(@($jobManifest, "hu_m43_fold_job_manifest_v1", "pass"), @($done, "hu_m43_fold_job_done_v1", "complete"))) {
        $object = $check[0]
        if ($object.schema -ne $check[1] -or $object.status -ne $check[2] -or $object.run_name -ne $RunName -or
            (Get-StrictInteger $object.job_index "$($check[1]).job_index") -ne $i -or [string]$object.job_kind -ne [string]$expected.job_kind -or
            (Get-StrictInteger $object.outer_fold "$($check[1]).outer_fold") -ne (Get-StrictInteger $expected.outer_fold "manifest.outer_fold") -or
            (($null -eq $expected.inner_fold -and $null -ne $object.inner_fold) -or
             ($null -ne $expected.inner_fold -and (Get-StrictInteger $object.inner_fold "$($check[1]).inner_fold") -ne (Get-StrictInteger $expected.inner_fold "manifest.inner_fold"))) -or
            [string]$object.job_spec_sha256 -ne [string]$expected.job_spec_sha256 -or
            [string]$object.source_sha256 -ne [string]$manifest.source.sha256 -or
            [string]$object.run_manifest_sha256 -ne $manifestSha -or
            [string]$object.cloud_contract_sha256 -ne [string]$manifest.cloud_contract.file_sha256 -or
            [string]$object.input_bundle_sha256 -ne [string]$manifest.inputs.input_bundle_sha256 -or
            $object.current_profile_mutated -ne $false -or $object.no_runtime_activation -ne $true) {
            throw "Downloaded M4.3 job object is stale or invalid at job $i"
        }
    }
    $artifactSha = Get-Sha256 $estimatorPath
    $jobManifestSha = Get-Sha256 $jobManifestPath
    if ($artifactSha -ne [string]$jobManifest.artifact_sha256 -or $artifactSha -ne [string]$done.artifact_sha256 -or
        $jobManifestSha -ne [string]$done.job_manifest_sha256) {
        throw "Downloaded M4.3 job hash chain is invalid at job $i"
    }
    $verifiedJobs += [ordered]@{
        job_index = $i; job_kind = [string]$expected.job_kind; outer_fold = Get-StrictInteger $expected.outer_fold "manifest.outer_fold"
        inner_fold = $expected.inner_fold; job_spec_sha256 = [string]$expected.job_spec_sha256
        artifact_sha256 = $artifactSha; job_manifest_sha256 = $jobManifestSha
    }
}

$outputModel = Join-Path $stageDir "model.pkl"
$trainingManifestOutput = Join-Path $stageDir "training_manifest.json"
$assemblerStdout = Join-Path $stageDir "assembler_stdout.log"
$assemblerStderr = Join-Path $stageDir "assembler_stderr.log"
$hp = $manifest.training_config
$thresholdArgument = (@($hp.thresholds | ForEach-Object { [string]::Format([Globalization.CultureInfo]::InvariantCulture, "{0:G17}", [double]$_) }) -join ",")
$assemblerArgs = @("-m", "ofc_regular.assemble_hu_m43_fold_training", "--fold-artifacts-dir", $foldArtifactsDir)
foreach ($path in $resolvedTrain) { $assemblerArgs += @("--train", $path) }
foreach ($path in $resolvedCalibration) { $assemblerArgs += @("--calibration", $path) }
$assemblerArgs += @(
    "--m43-data-contract", $resolvedContract, "--m43-plan", $resolvedPlan,
    "--fold-cloud-contract", $cloudContractPath, "--repo-root", $repoRoot,
    "--run-manifest", $manifestPath, "--predeclared-receipt", $predeclaredReceipt,
    "--local-sealed-rebind-manifest", $localRebindPath, "--source-archive", $frozenSourcePath,
    "--output-model", $outputModel, "--manifest-output", $trainingManifestOutput,
    "--action-score-mode", "$($hp.action_score_mode)", "--model-id", "$($hp.model_id)",
    "--cross-fit-folds", "$($hp.cross_fit_folds)", "--iterations", "$($hp.iterations)",
    "--max-leaf-nodes", "$($hp.max_leaf_nodes)", "--learning-rate", "$($hp.learning_rate)",
    "--l2-regularization", "$($hp.l2_regularization)", "--seed", "$($hp.seed)",
    "--paired-se-floor", "$($hp.paired_se_floor)", "--paired-huber-alpha", "$($hp.paired_huber_alpha)",
    "--downside-quantile", "$($hp.downside_quantile)", "--positive-gain-score-weight", "$($hp.positive_gain_score_weight)",
    "--downside-risk-score-weight", "$($hp.downside_risk_score_weight)",
    "--ensemble-disagreement-score-weight", "$($hp.ensemble_disagreement_score_weight)",
    "--near-best-margin", "$($hp.near_best_margin)", "--minimum-safe-teacher-gain", "$($hp.minimum_safe_teacher_gain)",
    "--safety-calibrator-c", "$($hp.safety_calibrator_c)", "--safety-fit-ratio", "$($hp.safety_fit_ratio)",
    "--safety-split-seed", "$($hp.safety_split_seed)", "--minimum-safety-fit-samples", "$($hp.minimum_safety_fit_samples)",
    "--minimum-threshold-lock-samples", "$($hp.minimum_threshold_lock_samples)", "--thresholds", $thresholdArgument,
    "--minimum-calibration-fires", "$($hp.minimum_calibration_fires)",
    "--maximum-false-positive-rate", "$($hp.maximum_false_positive_rate)", "--maximum-p95-loss", "$($hp.maximum_p95_loss)",
    "--maximum-p99-loss", "$($hp.maximum_p99_loss)", "--maximum-max-loss", "$($hp.maximum_max_loss)", "--run-name", $RunName,
    "--run-manifest-sha256", $manifestSha, "--source-sha256", "$($manifest.source.sha256)"
)
$deterministicEnvironment = [ordered]@{ PYTHONPATH = $frozenPythonPath; PYTHONHASHSEED = "0"; OMP_NUM_THREADS = "1"; OPENBLAS_NUM_THREADS = "1"; MKL_NUM_THREADS = "1"; NUMEXPR_NUM_THREADS = "1" }
$oldEnvironment = @{}
foreach ($name in $deterministicEnvironment.Keys) {
    $oldEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
    [Environment]::SetEnvironmentVariable($name, [string]$deterministicEnvironment[$name], "Process")
}
$oldPreference = $ErrorActionPreference; $ErrorActionPreference = "Continue"
$importPreflight = @'
import pathlib,sys
root=pathlib.Path(sys.argv[1]).resolve()
import ofc_regular
import ofc_regular.assemble_hu_m43_fold_training as assembler
import ofc_regular.train_hu_m4_joint_model as trainer
import numpy,sklearn
for module in (ofc_regular,assembler,trainer):
    pathlib.Path(module.__file__).resolve().relative_to(root)
assert numpy.__version__=="2.2.6" and sklearn.__version__=="1.8.0"
'@
try {
    $importOutput = @($importPreflight | & python - $frozenPythonPath 2>&1)
    if ($LASTEXITCODE -ne 0) { throw "Frozen source import/dependency preflight failed: $($importOutput -join [Environment]::NewLine)" }
    & python @assemblerArgs 1> $assemblerStdout 2> $assemblerStderr
    $assemblerCode = $LASTEXITCODE
}
finally {
    foreach ($name in $oldEnvironment.Keys) { [Environment]::SetEnvironmentVariable($name, $oldEnvironment[$name], "Process") }
    $ErrorActionPreference = $oldPreference
}
if ($assemblerCode -ne 0) { throw "M4.3 local assembler failed once with exit $assemblerCode; staging preserved at $stageDir" }
if (-not (Test-Path $outputModel -PathType Leaf) -or -not (Test-Path $trainingManifestOutput -PathType Leaf)) {
    throw "M4.3 assembler did not produce both final artifacts"
}
$trainingManifest = Get-Content -LiteralPath $trainingManifestOutput -Raw | ConvertFrom-Json
$modelSha = Get-Sha256 $outputModel
$trainingManifestSha = Get-Sha256 $trainingManifestOutput
$assembly = $trainingManifest.distributed_fold_assembly
if ($trainingManifest.schema -ne "hu_m4_t1_joint_training_manifest_v2" -or $assembly.schema -ne "hu_m43_fold_assembly_v1" -or
    $assembly.status -ne "pass" -or (Get-StrictInteger $assembly.job_count "assembly.job_count") -ne $ExpectedJobs -or
    [string]$assembly.cloud_contract_sha256 -ne [string]$manifest.cloud_contract.file_sha256 -or
    [string]$assembly.source_sha256 -ne [string]$manifest.source.sha256 -or @($assembly.jobs).Count -ne $ExpectedJobs -or
    [string]$trainingManifest.runtime_lock.candidate_model_sha256 -ne $modelSha -or
    [string]$trainingManifest.runtime_lock.safety_model_sha256 -ne $modelSha -or
    $trainingManifest.locked_holdout_used_for_threshold_or_training -ne $false -or
    $trainingManifest.current_profile_mutated -eq $true) {
    throw "Assembled M4.3 training manifest violates the frozen distributed contract"
}
for ($i = 0; $i -lt $ExpectedJobs; $i++) {
    $actual = $assembly.jobs[$i]; $expected = $verifiedJobs[$i]
    if ((Get-StrictInteger $actual.job_index "assembly.jobs[$i].job_index") -ne $i -or [string]$actual.artifact_sha256 -ne [string]$expected.artifact_sha256 -or
        [string]$actual.job_manifest_sha256 -ne [string]$expected.job_manifest_sha256 -or
        [string]$actual.job_spec_sha256 -ne [string]$expected.job_spec_sha256) {
        throw "Assembler job binding mismatch at job $i"
    }
}

Copy-Item -LiteralPath $cloudContractPath -Destination (Join-Path $stageDir "fold_cloud_contract.json")
$localRebindCopy = Join-Path $stageDir "m43_local_sealed_rebind_manifest.json"
Copy-Item -LiteralPath $localRebindPath -Destination $localRebindCopy
$localRebind = Get-Content -LiteralPath $localRebindPath -Raw | ConvertFrom-Json
$receipt = [ordered]@{
    schema = "hu_m43_fold_model_receipt_v1"; status = "verified"; run_name = $RunName
    run_manifest_sha256 = $manifestSha; source_sha256 = [string]$manifest.source.sha256
    cloud_contract_sha256 = [string]$manifest.cloud_contract.file_sha256
    local_sealed_rebind_manifest_file_sha256 = Get-Sha256 $localRebindPath
    local_sealed_rebind_sha256 = [string]$localRebind.rebind_sha256
    job_count = $ExpectedJobs; jobs = $verifiedJobs; model_sha256 = $modelSha; training_manifest_sha256 = $trainingManifestSha
    current_profile_mutated = $false; no_runtime_activation = $true; received_at = (Get-Date).ToUniversalTime().ToString("o")
}
$receiptPath = Join-Path $stageDir "receipt.json"
[IO.File]::WriteAllText($receiptPath, (($receipt | ConvertTo-Json -Depth 15) + "`n"), [Text.UTF8Encoding]::new($false))
if (Test-Path -LiteralPath $OutputDir) { throw "OutputDir appeared during receive; refusing overwrite. Staging preserved at $stageDir" }
Move-Item -LiteralPath $stageDir -Destination $OutputDir
$receipt | ConvertTo-Json -Depth 15
