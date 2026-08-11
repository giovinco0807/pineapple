param(
    [Parameter(Mandatory = $true)][string]$RunName,
    [string]$ProjectId = "ofc-solver-485418",
    [string]$Bucket = "pokerhu-ofc-solver-485418-training"
)

$ErrorActionPreference = "Stop"
$ExpectedJobs = 30
$JobsPerShard = 4

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

function Invoke-Gcloud {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    try { $output = & gcloud @Arguments 2>&1; $code = $LASTEXITCODE }
    finally { $ErrorActionPreference = $old }
    if ($code -ne 0) { throw "gcloud failed ($code): gcloud $($Arguments -join ' ')`n$(@($output) -join [Environment]::NewLine)" }
    return @($output)
}

function Get-GcsJson {
    param([Parameter(Mandatory = $true)][string]$Uri)
    $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    try { $raw = & gcloud storage cat $Uri 2>&1; $code = $LASTEXITCODE }
    finally { $ErrorActionPreference = $old }
    if ($code -eq 0) { return (@($raw) -join "`n") | ConvertFrom-Json }
    $message = @($raw) -join [Environment]::NewLine
    if ($message -match '(?i)not found|does not exist|No URLs matched|matched no objects|404') { return $null }
    throw "Unable to read GCS JSON: $Uri`n$message"
}

function Get-GcsUris {
    param([Parameter(Mandatory = $true)][string]$Pattern)
    $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    try { $raw = & gcloud storage ls $Pattern 2>&1; $code = $LASTEXITCODE }
    finally { $ErrorActionPreference = $old }
    if ($code -eq 0) { return @($raw | ForEach-Object { ([string]$_).Trim() } | Where-Object { $_ }) }
    $message = @($raw) -join [Environment]::NewLine
    if ($message -match '(?i)not found|does not exist|No URLs matched|matched no objects|404') { return @() }
    throw "Unable to list GCS status objects: $Pattern`n$message"
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
        (Get-StrictInteger $policy.module_count "source_package_policy.module_count") -lt 3 -or [string]$policy.entries_sha256 -notmatch '^[0-9a-f]{64}$') {
        throw "Frozen source package policy changed"
    }
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
        (Get-StrictInteger $Manifest.compute.jobs_per_shard "compute.jobs_per_shard") -ne $JobsPerShard -or $Manifest.compute.auto_delete -ne $true) {
        throw "Local M4.3 fold manifest identity/schema is invalid"
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
            [string]$job.job_kind -ne $kind -or $outerFold -ne $outer -or $shardIndex -ne [Math]::Floor($i / $JobsPerShard) -or
            (($null -eq $inner -and $null -ne $job.inner_fold) -or ($null -ne $inner -and $innerFold -ne $inner)) -or
            (Get-StrictInteger $jobSpec.job_index "job_spec.job_index") -ne $i -or [string]$jobSpec.kind -ne $kind -or
            (Get-StrictInteger $jobSpec.outer_fold "job_spec.outer_fold") -ne $outer -or
            (($null -eq $inner -and $null -ne $jobSpec.inner_fold) -or ($null -ne $inner -and $specInner -ne $inner)) -or
            [string]$job.done_uri -ne ("gs://$Bucket/runs/$RunName/results/job-{0:D2}/DONE.json" -f $i)) {
            throw "M4.3 manifest job mapping is invalid at $i"
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
        throw "Frozen full training configuration changed"
    }
}

function Assert-JobObject {
    param(
        [Parameter(Mandatory = $true)]$Object,
        [Parameter(Mandatory = $true)][string]$Schema,
        [Parameter(Mandatory = $true)][int]$UriJob,
        [Parameter(Mandatory = $true)]$Manifest,
        [Parameter(Mandatory = $true)][string]$ManifestSha
    )
    $expected = $Manifest.jobs[$UriJob]
    $objectJob = Get-StrictInteger $Object.job_index "$Schema.job_index"
    $objectOuter = Get-StrictInteger $Object.outer_fold "$Schema.outer_fold"
    $objectInner = if ($null -eq $Object.inner_fold) { $null } else { Get-StrictInteger $Object.inner_fold "$Schema.inner_fold" }
    if ($Object.schema -ne $Schema -or $Object.run_name -ne $RunName -or $objectJob -ne $UriJob -or
        [string]$Object.job_kind -ne [string]$expected.job_kind -or $objectOuter -ne (Get-StrictInteger $expected.outer_fold "manifest.outer_fold") -or
        (($null -eq $expected.inner_fold -and $null -ne $Object.inner_fold) -or
         ($null -ne $expected.inner_fold -and $objectInner -ne (Get-StrictInteger $expected.inner_fold "manifest.inner_fold"))) -or
        [string]$Object.job_spec_sha256 -ne [string]$expected.job_spec_sha256 -or
        [string]$Object.run_manifest_sha256 -ne $ManifestSha -or
        [string]$Object.source_sha256 -ne [string]$Manifest.source.sha256 -or
        [string]$Object.cloud_contract_sha256 -ne [string]$Manifest.cloud_contract.file_sha256 -or
        [string]$Object.input_bundle_sha256 -ne [string]$Manifest.inputs.input_bundle_sha256 -or
        $Object.current_profile_mutated -ne $false -or $Object.no_runtime_activation -ne $true) {
        throw "Stale or invalid $Schema object for URI job $UriJob"
    }
}

if ($RunName -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$') { throw "RunName is not path-safe" }
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$manifestPath = Join-Path $repoRoot "outputs/gcp_runs/$RunName/m43_fold_run_manifest.json"
$cloudContractPath = Join-Path $repoRoot "outputs/gcp_runs/$RunName/fold_cloud_contract.json"
$sourceArchivePath = Join-Path $repoRoot "outputs/gcp_runs/$RunName/ofc_regular_hu_m43_fold_source.zip"
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) { throw "Local frozen M4.3 fold manifest is missing: $manifestPath" }
if (-not (Test-Path -LiteralPath $cloudContractPath -PathType Leaf)) { throw "Local frozen cloud contract is missing: $cloudContractPath" }
if (-not (Test-Path -LiteralPath $sourceArchivePath -PathType Leaf)) { throw "Local frozen source archive is missing: $sourceArchivePath" }
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$manifestSha = Get-Sha256 $manifestPath
Assert-Manifest $manifest
if ((Get-Sha256 $sourceArchivePath) -ne [string]$manifest.source.sha256 -or
    [long](Get-Item -LiteralPath $sourceArchivePath).Length -ne (Get-StrictInteger $manifest.source.bytes "source.bytes")) { throw "Local frozen source archive hash/size mismatch" }
Assert-SourcePackagePolicy -Manifest $manifest -ArchivePath $sourceArchivePath
$manifestRaw = Get-Content -LiteralPath $manifestPath -Raw
$projectionRaw = Get-Content -LiteralPath $cloudContractPath -Raw
if ($manifestRaw -match '(?i)locked' -or $projectionRaw -match '(?i)locked' -or
    (Get-Sha256 $cloudContractPath) -ne [string]$manifest.cloud_contract.file_sha256) {
    throw "Local frozen cloud state violates the redaction/hash boundary"
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
assert m["cloud_contract"]["predeclared_training_receipt_sha256"]==p["predeclared_training_receipt_sha256"]=="9313bdc6ffa33335032ee3d98e279d123d2f3aa70571114e75ee5f38e2594bea"
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
if ($configCode -ne 0) { throw "Local run/projection configuration binding is invalid: $($configOutput -join [Environment]::NewLine)" }

$prefix = "gs://$Bucket/runs/$RunName/results"
$donePattern = "^" + [regex]::Escape($prefix) + "/job-(\d{2})/DONE\.json$"
$seenDoneJobs = [Collections.Generic.HashSet[int]]::new()
$doneByJob = [ordered]@{}
foreach ($uri in @(Get-GcsUris "$prefix/**/DONE.json")) {
    $match = [regex]::Match($uri, $donePattern)
    if (-not $match.Success) { throw "Unexpected M4.3 DONE URI: $uri" }
    $jobIndex = [int]$match.Groups[1].Value
    if ($jobIndex -lt 0 -or $jobIndex -ge $ExpectedJobs) { throw "DONE URI job is out of range: $uri" }
    if (-not $seenDoneJobs.Add($jobIndex)) { throw "Duplicate DONE job identity: $jobIndex" }
    if ($uri -ne [string]$manifest.jobs[$jobIndex].done_uri) { throw "DONE URI does not match frozen job mapping: $uri" }
    $done = Get-GcsJson $uri
    if ($null -eq $done -or $done.status -ne "complete") { throw "Listed DONE object is unreadable or incomplete: $uri" }
    Assert-JobObject $done "hu_m43_fold_job_done_v1" $jobIndex $manifest $manifestSha
    if ([string]$done.artifact_sha256 -notmatch '^[0-9a-f]{64}$' -or [string]$done.job_manifest_sha256 -notmatch '^[0-9a-f]{64}$') {
        throw "DONE output hashes are invalid for job $jobIndex"
    }
    $doneByJob["$jobIndex"] = $done
}

$statusByJob = [ordered]@{}
$heartbeatByJob = [ordered]@{}
foreach ($type in @("status", "heartbeat")) {
    $pattern = "^" + [regex]::Escape($prefix) + "/job-(\d{2})/$type\.json$"
    $seen = [Collections.Generic.HashSet[int]]::new()
    foreach ($uri in @(Get-GcsUris "$prefix/**/$type.json")) {
        $match = [regex]::Match($uri, $pattern)
        if (-not $match.Success) { throw "Unexpected M4.3 $type URI: $uri" }
        $jobIndex = [int]$match.Groups[1].Value
        if ($jobIndex -lt 0 -or $jobIndex -ge $ExpectedJobs -or -not $seen.Add($jobIndex)) { throw "Invalid/duplicate $type job identity: $uri" }
        $object = Get-GcsJson $uri
        if ($null -eq $object -or $object.run_name -ne $RunName -or (Get-StrictInteger $object.job_index "$type.job_index") -ne $jobIndex -or
            [string]$object.run_manifest_sha256 -ne $manifestSha -or [string]$object.source_sha256 -ne [string]$manifest.source.sha256) {
            throw "Stale or invalid M4.3 $type object: $uri"
        }
        if ($type -eq "status") {
            if ($object.schema -ne "hu_m43_fold_job_status_v1") { throw "Invalid status schema: $uri" }
            $statusByJob["$jobIndex"] = $object
        }
        else {
            if ($object.schema -ne "hu_m43_fold_job_heartbeat_v1") { throw "Invalid heartbeat schema: $uri" }
            $heartbeatByJob["$jobIndex"] = $object
        }
    }
}

$missingJobs = @(0..($ExpectedJobs - 1) | Where-Object { -not $seenDoneJobs.Contains($_) })
$resumeShards = @($missingJobs | ForEach-Object { [Math]::Floor($_ / $JobsPerShard) } | Sort-Object -Unique)
$vmPrefix = [string]$manifest.compute.vm_prefix
$instanceRaw = Invoke-Gcloud @("compute", "instances", "list", "--project", $ProjectId, "--filter", ("name~'^{0}-g[0-9]{{2}}$'" -f [regex]::Escape($vmPrefix)), "--format=json")
$instances = if ($instanceRaw) { @(((@($instanceRaw) -join "`n") | ConvertFrom-Json)) } else { @() }
foreach ($instance in $instances) {
    if ([string]$instance.name -notmatch ("^" + [regex]::Escape($vmPrefix) + "-g0[0-7]$")) { throw "Unexpected VM returned for frozen run: $($instance.name)" }
}

[ordered]@{
    schema = "hu_m43_fold_spot_status_report_v1"; run_name = $RunName
    state = if ($seenDoneJobs.Count -eq $ExpectedJobs) { "complete" } elseif ($instances.Count -gt 0) { "running" } else { "incomplete" }
    manifest_sha256 = $manifestSha; completed_jobs = $seenDoneJobs.Count; total_jobs = $ExpectedJobs
    completed_job_indices = @($seenDoneJobs | Sort-Object); missing_job_indices = $missingJobs; resume_shards = $resumeShards
    resume_command = if ($resumeShards.Count) { ".\scripts\Start-GcpHuM43ModelRun.ps1 -RunName '$RunName' -StartShards $($resumeShards -join ',') -ResumeExisting -CreateInstances" } else { $null }
    done = $doneByJob; status = $statusByJob; heartbeat = $heartbeatByJob
    active_instances = $instances.Count
    instances = @($instances | ForEach-Object { [ordered]@{ name = $_.name; zone = ([string]$_.zone).Split('/')[-1]; status = $_.status; machine_type = ([string]$_.machineType).Split('/')[-1] } })
    current_profile_mutated = $false; no_runtime_activation = $true
} | ConvertTo-Json -Depth 15
