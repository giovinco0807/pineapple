param(
    [Parameter(Mandatory = $true)][string]$RunName,
    [string]$ProjectId = "ofc-solver-485418",
    [string]$Bucket = "pokerhu-ofc-solver-485418-training",
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"
$ManifestSchema = "hu_m43_attempt02_population_spot_manifest_v1"
$DoneSchema = "hu_m43_attempt02_population_spot_done_v1"
$ReceiptSchema = "hu_m43_attempt02_population_spot_receipt_v1"
if ($RunName -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$') { throw "RunName contains unsafe characters" }

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}
function Invoke-Gcloud([string[]]$Arguments) {
    $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    try { $output = & gcloud @Arguments 2>&1; $code = $LASTEXITCODE }
    finally { $ErrorActionPreference = $old }
    if ($code -ne 0) { throw "gcloud failed ($code): $(@($output) -join [Environment]::NewLine)" }
    return @($output)
}
function Write-Utf8NoBom([string]$Path, [string]$Text) {
    [System.IO.File]::WriteAllText($Path, $Text, [System.Text.UTF8Encoding]::new($false))
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$manifestPath = Join-Path $repoRoot "outputs/gcp_runs/$RunName/population_run_manifest.json"
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) { throw "Frozen Attempt02 population manifest is missing: $manifestPath" }
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$manifestSha = Get-Sha256 $manifestPath
if ($manifest.schema -ne $ManifestSchema -or $manifest.run_name -ne $RunName -or
    $manifest.project_id -ne $ProjectId -or $manifest.bucket -ne $Bucket -or
    $manifest.runtime.model_schema -ne "hu_m43_t1_joint_model_v4" -or
    $manifest.runtime.action_score_mode -ne "centered_fold_mean_delta_then_safety_gate_v4" -or
    $manifest.source_boundary.teacher_jsonl_packaged -ne $false -or
    $manifest.source_boundary.calibration_jsonl_packaged -ne $false -or
    $manifest.source_boundary.locked_jsonl_packaged -ne $false -or
    $manifest.source_boundary.current_profile_artifact_packaged -ne $false -or
    $manifest.no_runtime_activation -ne $true -or $manifest.current_profile_mutated -ne $false) {
    throw "Frozen Attempt02 population manifest identity/boundary mismatch"
}

$root = [System.IO.Path]::GetFullPath((Join-Path $repoRoot "outputs/hu_joint_policy/m43_attempt02_population"))
if ($OutputDir) {
    if (-not [System.IO.Path]::IsPathRooted($OutputDir)) { $OutputDir = Join-Path $repoRoot $OutputDir }
    $OutputDir = [System.IO.Path]::GetFullPath($OutputDir)
}
else { $OutputDir = [System.IO.Path]::GetFullPath((Join-Path $root $RunName)) }
$rootPrefix = $root.TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
if (-not $OutputDir.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "OutputDir must stay under outputs/hu_joint_policy/m43_attempt02_population; refusing profile/current overwrite"
}
if (Test-Path -LiteralPath $OutputDir) {
    $receiptPath = Join-Path $OutputDir "receipt.json"
    if (-not (Test-Path -LiteralPath $receiptPath -PathType Leaf)) { throw "Attempt02 population OutputDir exists without a receipt; refusing overwrite" }
    $receipt = Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json
    if ($receipt.schema -ne $ReceiptSchema -or $receipt.status -ne "verified_and_merged" -or
        $receipt.run_name -ne $RunName -or $receipt.run_manifest_sha256 -ne $manifestSha -or
        $receipt.population_plan_sha256 -ne $manifest.population_plan.sha256 -or
        $receipt.model_sha256 -ne $manifest.runtime.model_sha256 -or
        (Get-Sha256 (Join-Path $OutputDir "evaluation.json")) -ne $receipt.evaluation_sha256 -or
        (Get-Sha256 (Join-Path $OutputDir "records.jsonl")) -ne $receipt.records_sha256 -or
        (Get-Sha256 (Join-Path $OutputDir "merge_manifest.json")) -ne $receipt.merge_manifest_sha256 -or
        (Get-Sha256 (Join-Path $OutputDir "run_manifest.json")) -ne $receipt.run_manifest_sha256 -or
        $receipt.current_profile_mutated -ne $false -or $receipt.no_runtime_activation -ne $true) {
        throw "Existing Attempt02 population receipt/hash chain is invalid"
    }
    $receipt | ConvertTo-Json -Depth 12
    exit 0
}

New-Item -ItemType Directory -Path $root -Force | Out-Null
$stage = Join-Path $root (".receiving-{0}-{1}" -f $RunName, [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $stage | Out-Null
$prefix = "gs://$Bucket/runs/$RunName"
$planPath = Join-Path $stage "population_plan.json"
Copy-Item -LiteralPath $manifestPath -Destination (Join-Path $stage "run_manifest.json")
Invoke-Gcloud @("storage", "cp", "$prefix/source/population_plan.json", $planPath, "--project", $ProjectId)
if ((Get-Sha256 $planPath) -ne [string]$manifest.population_plan.sha256) { throw "Downloaded Attempt02 population plan hash mismatch; staging preserved at $stage" }

$evaluationPaths = @(); $recordPaths = @(); $shardReceipts = @()
for ($shard = 0; $shard -lt [int]$manifest.shards.count; $shard++) {
    $name = "shard-{0:D4}" -f $shard
    $dir = Join-Path $stage $name
    New-Item -ItemType Directory -Path $dir | Out-Null
    foreach ($file in @("DONE", "evaluation.json", "records.jsonl")) {
        Invoke-Gcloud @("storage", "cp", "$prefix/results/$name/$file", (Join-Path $dir $file), "--project", $ProjectId)
    }
    $donePath = Join-Path $dir "DONE"
    $evaluationPath = Join-Path $dir "evaluation.json"
    $recordsPath = Join-Path $dir "records.jsonl"
    $done = Get-Content -LiteralPath $donePath -Raw | ConvertFrom-Json
    if ($done.schema -ne $DoneSchema -or $done.status -ne "complete" -or
        $done.run_name -ne $RunName -or [int]$done.shard -ne $shard -or
        $done.manifest_sha256 -ne $manifestSha -or $done.source_sha256 -ne $manifest.source.sha256 -or
        $done.model_sha256 -ne $manifest.runtime.model_sha256 -or
        $done.evaluation_sha256 -ne (Get-Sha256 $evaluationPath) -or
        $done.records_sha256 -ne (Get-Sha256 $recordsPath) -or
        $done.current_profile_mutated -ne $false -or $done.no_runtime_activation -ne $true) {
        throw "Attempt02 population shard DONE/hash chain mismatch: $shard; staging preserved at $stage"
    }
    $evaluationPaths += $evaluationPath
    $recordPaths += $recordsPath
    $shardReceipts += [ordered]@{ shard = $shard; done_sha256 = Get-Sha256 $donePath; evaluation_sha256 = $done.evaluation_sha256; records_sha256 = $done.records_sha256 }
}

$evaluationOutput = Join-Path $stage "evaluation.json"
$recordsOutput = Join-Path $stage "records.jsonl"
$mergeOutput = Join-Path $stage "merge_manifest.json"
$mergeArgs = @("-m", "ofc_regular.merge_hu_m4_population_shards", "--plan", $planPath)
foreach ($path in $evaluationPaths) { $mergeArgs += @("--shard-evaluation", $path) }
foreach ($path in $recordPaths) { $mergeArgs += @("--shard-records", $path) }
$mergeArgs += @("--records-output", $recordsOutput, "--output", $evaluationOutput, "--merge-manifest-output", $mergeOutput)
$oldPythonPath = $env:PYTHONPATH; $env:PYTHONPATH = Join-Path $repoRoot "src"
try { & python @mergeArgs; if ($LASTEXITCODE -ne 0) { throw "Attempt02 population shard merger failed ($LASTEXITCODE); staging preserved at $stage" } }
finally { $env:PYTHONPATH = $oldPythonPath }

$evaluation = Get-Content -LiteralPath $evaluationOutput -Raw | ConvertFrom-Json
$merge = Get-Content -LiteralPath $mergeOutput -Raw | ConvertFrom-Json
$runtime = $evaluation.runtime_config
if ($evaluation.schema -ne "hu_m4_t1_population_evaluation_v1" -or
    [int]$evaluation.paired_seeds_per_opponent -ne [int]$manifest.population_plan.paired_seeds -or
    $runtime.candidate_model_sha256 -ne $manifest.runtime.model_sha256 -or
    $runtime.safety_model_sha256 -ne $manifest.runtime.model_sha256 -or
    $runtime.model_schema -ne "hu_m43_t1_joint_model_v4" -or
    $runtime.model_id -ne $manifest.runtime.model_id -or
    $runtime.action_score_mode -ne "centered_fold_mean_delta_then_safety_gate_v4" -or
    $runtime.runtime_binding_verified -ne $true -or
    $runtime.freeze_manifest_sha256 -ne $manifest.runtime.freeze_manifest_sha256 -or
    $runtime.training_manifest_sha256 -ne $manifest.runtime.training_manifest_sha256 -or
    $runtime.current_profile_used -ne $false -or $runtime.promotion_artifact_contract -ne $true -or
    $merge.status -ne "complete_content_verified" -or
    $merge.population_plan_sha256 -ne $manifest.population_plan.sha256) {
    throw "Merged Attempt02 population violates the frozen v4 runtime/plan contract; staging preserved at $stage"
}

$receipt = [ordered]@{
    schema = $ReceiptSchema
    status = "verified_and_merged"
    run_name = $RunName
    run_manifest_sha256 = $manifestSha
    population_plan_sha256 = $manifest.population_plan.sha256
    model_sha256 = $manifest.runtime.model_sha256
    model_schema = "hu_m43_t1_joint_model_v4"
    action_score_mode = "centered_fold_mean_delta_then_safety_gate_v4"
    evaluation_path = [System.IO.Path]::GetFullPath((Join-Path $OutputDir "evaluation.json"))
    evaluation_sha256 = Get-Sha256 $evaluationOutput
    records_path = [System.IO.Path]::GetFullPath((Join-Path $OutputDir "records.jsonl"))
    records_sha256 = Get-Sha256 $recordsOutput
    merge_manifest_sha256 = Get-Sha256 $mergeOutput
    paired_seeds_per_opponent = [int]$evaluation.paired_seeds_per_opponent
    valid_overrides = [int]$evaluation.population.all_seats.overrides
    invalid_counterfactuals = [int]$evaluation.invalid_counterfactuals
    nonfire_cancellation_mismatches = [int]$evaluation.nonfire_cancellation_mismatches
    shards = $shardReceipts
    teacher_calibration_locked_content_received = $false
    current_profile_mutated = $false
    no_runtime_activation = $true
    received_at = (Get-Date).ToUniversalTime().ToString("o")
}
Write-Utf8NoBom (Join-Path $stage "receipt.json") (($receipt | ConvertTo-Json -Depth 12) + "`n")
if (Test-Path -LiteralPath $OutputDir) { throw "Attempt02 OutputDir appeared during receive; refusing overwrite. Staging preserved at $stage" }
Move-Item -LiteralPath $stage -Destination $OutputDir
$receipt | ConvertTo-Json -Depth 12
