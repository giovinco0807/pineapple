param(
    [Parameter(Mandatory = $true)][string]$RunName,
    [string]$ProjectId = "ofc-solver-485418",
    [string]$Bucket = "pokerhu-ofc-solver-485418-training",
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"
$ManifestSchema = "hu_m43_attempt12_population_spot_manifest_v1"
$DoneSchema = "hu_m43_attempt12_population_spot_done_v1"
$ReceiptSchema = "hu_m43_attempt12_population_spot_receipt_v1"
$AcceptanceStatusSchema = "hu_m43_attempt12_population_acceptance_status_v1"
$PopulationPlanSha256 = "66d9c3dc50bcd6ae2ce72e2a558dcfae5720d3cb2387a16ef810d1498f4db462"
$SeedRegistrySha256 = "de63af1ded1bfe6ff7c92dc6bcc8aedec9c408956e687c856198143bffca8255"
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
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) { throw "Frozen Attempt12 population manifest is missing: $manifestPath" }
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$manifestSha = Get-Sha256 $manifestPath
if ($manifest.schema -ne $ManifestSchema -or $manifest.run_name -ne $RunName -or
    $manifest.project_id -ne $ProjectId -or $manifest.bucket -ne $Bucket -or
    [int]$manifest.shards.count -ne 20 -or
    [int]$manifest.population_plan.paired_seeds -ne 1000 -or
    [long]$manifest.population_plan.seed -ne 220108071901 -or
    [long]$manifest.population_plan.seed_stride -ne 1000003 -or
    [int]$manifest.population_plan.paired_seeds_per_shard -ne 50 -or
    $manifest.runtime.model_schema -ne "hu_m43_attempt12_t1_second_distilled_selector_v1" -or
    $manifest.runtime.artifact_schema -ne "hu_m43_attempt12_t1_second_distilled_pickle_v1" -or
    $manifest.runtime.feature_schema -ne "hu_m43_attempt12_lambda_all_legal_public_infoset_features_v1" -or
    $manifest.runtime.head_schema -ne "hu_m43_attempt12_policy_delta_safe_tail_heads_v1" -or
    $manifest.runtime.action_score_mode -ne "attempt12_lambda_all_legal_distilled_safe_selector_v1" -or
    $manifest.runtime.baseline_profile -ne "stage19_p0" -or
    $manifest.runtime.source_model_manifest_sha256 -ne "e9b9d9748bb2b2f31a4baa0d928b60c05f254ab5d7915a7461fb8402e1e7ddb8" -or
    $manifest.runtime.source_native_manifest_sha256 -ne "ec295d070ac7bb21a82aeb2f432b8f6f191e61b63027bc446e8bd5f39f51569f" -or
    [string]$manifest.runtime.runtime_dependency_closure_sha256 -cnotmatch '^[0-9a-f]{64}$' -or
    $manifest.population_plan.sha256 -ne $PopulationPlanSha256 -or
    $manifest.runtime.seed_registry_sha256 -ne $SeedRegistrySha256 -or
    $manifest.launch_preflight.development200_full_fit_bound -ne $true -or
    $manifest.launch_preflight.audit50_one_shot_go_bound -ne $true -or
    [int]$manifest.launch_preflight.audit50_fit_rows -ne 0 -or
    $manifest.launch_preflight.threshold_reselection_performed -ne $false -or
    [string]$manifest.launch_preflight.development_decision_sha256 -cnotmatch '^[0-9a-f]{64}$' -or
    [string]$manifest.launch_preflight.development_selector_receipt_sha256 -cnotmatch '^[0-9a-f]{64}$' -or
    [string]$manifest.launch_preflight.development_pass_freeze_sha256 -cnotmatch '^[0-9a-f]{64}$' -or
    [string]$manifest.launch_preflight.audit50_decision_sha256 -cnotmatch '^[0-9a-f]{64}$' -or
    [string]$manifest.launch_preflight.audit50_selector_receipt_sha256 -cnotmatch '^[0-9a-f]{64}$' -or
    $manifest.checkpoint.unit -ne "completed_shard" -or
    $manifest.checkpoint.resume_missing_shards_only -ne $true -or
    $manifest.checkpoint.done_commit_last -ne $true -or
    $manifest.heartbeat.schema -ne "hu_m43_attempt12_population_spot_status_v1" -or
    $manifest.heartbeat.required -ne $true -or
    $manifest.acceptance_artifacts.model.sha256 -ne $manifest.runtime.model_sha256 -or
    $manifest.acceptance_artifacts.training_manifest.sha256 -ne $manifest.runtime.training_manifest_sha256 -or
    $manifest.acceptance_artifacts.runtime_freeze.sha256 -ne $manifest.runtime.runtime_freeze_sha256 -or
    $manifest.acceptance_artifacts.runtime_source_archive.sha256 -ne $manifest.runtime.runtime_source_archive_sha256 -or
    $manifest.acceptance_artifacts.runtime_source_manifest.sha256 -ne $manifest.runtime.runtime_source_manifest_sha256 -or
    $manifest.acceptance_artifacts.development_decision.sha256 -ne $manifest.launch_preflight.development_decision_sha256 -or
    $manifest.acceptance_artifacts.development_selector_receipt.sha256 -ne $manifest.launch_preflight.development_selector_receipt_sha256 -or
    $manifest.acceptance_artifacts.development_pass_freeze.sha256 -ne $manifest.launch_preflight.development_pass_freeze_sha256 -or
    $manifest.acceptance_artifacts.audit50_decision.sha256 -ne $manifest.launch_preflight.audit50_decision_sha256 -or
    $manifest.acceptance_artifacts.audit50_selector_receipt.sha256 -ne $manifest.launch_preflight.audit50_selector_receipt_sha256 -or
    $manifest.source_boundary.teacher_jsonl_packaged -ne $false -or
    $manifest.source_boundary.audit_jsonl_packaged -ne $false -or
    $manifest.source_boundary.calibration_jsonl_packaged -ne $false -or
    $manifest.source_boundary.locked_jsonl_packaged -ne $false -or
    $manifest.source_boundary.provenance_json_evidence_packaged -ne $true -or
    $manifest.source_boundary.current_profile_artifact_packaged -ne $false -or
    $manifest.source_boundary.runtime_artifacts_only -ne $true -or
    $manifest.no_runtime_activation -ne $true -or $manifest.current_profile_mutated -ne $false) {
    throw "Frozen Attempt12 population manifest identity/boundary mismatch"
}

$root = [System.IO.Path]::GetFullPath((Join-Path $repoRoot "outputs/hu_joint_policy/m43_attempt12_population"))
if ($OutputDir) {
    if (-not [System.IO.Path]::IsPathRooted($OutputDir)) { $OutputDir = Join-Path $repoRoot $OutputDir }
    $OutputDir = [System.IO.Path]::GetFullPath($OutputDir)
}
else { $OutputDir = [System.IO.Path]::GetFullPath((Join-Path $root $RunName)) }
$rootPrefix = $root.TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
if (-not $OutputDir.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "OutputDir must stay under outputs/hu_joint_policy/m43_attempt12_population; refusing profile/current overwrite"
}
if (Test-Path -LiteralPath $OutputDir) {
    $receiptPath = Join-Path $OutputDir "receipt.json"
    $statusPath = Join-Path $OutputDir "acceptance_status.json"
    if (-not (Test-Path -LiteralPath $receiptPath -PathType Leaf) -or
        -not (Test-Path -LiteralPath $statusPath -PathType Leaf)) {
        throw "Attempt12 population OutputDir exists without complete receipts; refusing overwrite"
    }
    $receipt = Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json
    $acceptance = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
    if ($receipt.schema -ne $ReceiptSchema -or $receipt.status -ne "verified_and_merged" -or
        $receipt.run_name -ne $RunName -or $receipt.run_manifest_sha256 -ne $manifestSha -or
        $receipt.population_plan_sha256 -ne $manifest.population_plan.sha256 -or
        $receipt.seed_registry_sha256 -ne $SeedRegistrySha256 -or
        $receipt.development_decision_sha256 -ne $manifest.launch_preflight.development_decision_sha256 -or
        $receipt.development_selector_receipt_sha256 -ne $manifest.launch_preflight.development_selector_receipt_sha256 -or
        $receipt.development_pass_freeze_sha256 -ne $manifest.launch_preflight.development_pass_freeze_sha256 -or
        $receipt.audit50_decision_sha256 -ne $manifest.launch_preflight.audit50_decision_sha256 -or
        $receipt.audit50_selector_receipt_sha256 -ne $manifest.launch_preflight.audit50_selector_receipt_sha256 -or
        $receipt.development200_full_fit_bound -ne $true -or
        $receipt.audit50_one_shot_go_bound -ne $true -or
        $receipt.model_sha256 -ne $manifest.runtime.model_sha256 -or
        (Get-Sha256 (Join-Path $OutputDir "evaluation.json")) -ne $receipt.evaluation_sha256 -or
        (Get-Sha256 (Join-Path $OutputDir "records.jsonl")) -ne $receipt.records_sha256 -or
        (Get-Sha256 (Join-Path $OutputDir "merge_manifest.json")) -ne $receipt.merge_manifest_sha256 -or
        (Get-Sha256 (Join-Path $OutputDir "run_manifest.json")) -ne $receipt.run_manifest_sha256 -or
        (Get-Sha256 (Join-Path $OutputDir "population_source.zip")) -ne $receipt.population_source_archive_sha256 -or
        (Get-Sha256 (Join-Path $OutputDir "runtime_source.zip")) -ne $receipt.runtime_source_archive_sha256 -or
        (Get-Sha256 (Join-Path $OutputDir "runtime_source_manifest.json")) -ne $receipt.runtime_source_manifest_sha256 -or
        (Get-Sha256 (Join-Path $OutputDir "development_decision.json")) -ne $receipt.development_decision_sha256 -or
        (Get-Sha256 (Join-Path $OutputDir "development_selector_receipt.json")) -ne $receipt.development_selector_receipt_sha256 -or
        (Get-Sha256 (Join-Path $OutputDir "development_pass_freeze.json")) -ne $receipt.development_pass_freeze_sha256 -or
        (Get-Sha256 (Join-Path $OutputDir "audit50_decision.json")) -ne $receipt.audit50_decision_sha256 -or
        (Get-Sha256 (Join-Path $OutputDir "audit50_selector_receipt.json")) -ne $receipt.audit50_selector_receipt_sha256 -or
        (Get-Sha256 $statusPath) -ne $receipt.acceptance_status_sha256 -or
        $acceptance.schema -ne $AcceptanceStatusSchema -or
        [string]$acceptance.status -notin @("complete_go", "complete_no_go") -or
        $acceptance.current_profile_mutated -ne $false -or
        $acceptance.runtime_policy_activated -ne $false -or
        $receipt.current_profile_mutated -ne $false -or $receipt.no_runtime_activation -ne $true) {
        throw "Existing Attempt12 population receipt/hash chain is invalid"
    }
    [ordered]@{ receipt = $receipt; acceptance = $acceptance } | ConvertTo-Json -Depth 15
    exit 0
}

New-Item -ItemType Directory -Path $root -Force | Out-Null
$stage = Join-Path $root (".receiving-{0}-{1}" -f $RunName, [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $stage | Out-Null
$prefix = "gs://$Bucket/runs/$RunName"
$planPath = Join-Path $stage "population_plan.json"
$runManifestCopy = Join-Path $stage "run_manifest.json"
Copy-Item -LiteralPath $manifestPath -Destination $runManifestCopy
Invoke-Gcloud @("storage", "cp", "$prefix/source/population_plan.json", $planPath, "--project", $ProjectId)
if ((Get-Sha256 $planPath) -ne $PopulationPlanSha256 -or
    [string]$manifest.population_plan.sha256 -ne $PopulationPlanSha256) { throw "Downloaded Attempt12 population plan hash mismatch; staging preserved at $stage" }
$populationSourceArchivePath = Join-Path $stage "population_source.zip"
$frozenPackageRoot = Join-Path $stage "frozen_package"
Invoke-Gcloud @("storage", "cp", [string]$manifest.source.uri, $populationSourceArchivePath, "--project", $ProjectId)
if ((Get-Sha256 $populationSourceArchivePath) -ne [string]$manifest.source.sha256) {
    throw "Downloaded Attempt12 frozen population source hash mismatch; staging preserved at $stage"
}
Expand-Archive -LiteralPath $populationSourceArchivePath -DestinationPath $frozenPackageRoot
if ((Get-Sha256 (Join-Path $frozenPackageRoot "artifacts/population_plan.json")) -ne [string]$manifest.population_plan.sha256) {
    throw "Frozen Attempt12 package plan hash mismatch; staging preserved at $stage"
}
$modelPath = Join-Path $stage "m43_model.pkl"
$trainingManifestPath = Join-Path $stage "training_manifest.json"
$runtimeFreezePath = Join-Path $stage "runtime_freeze.json"
$runtimeSourceArchivePath = Join-Path $stage "runtime_source.zip"
$runtimeSourceManifestPath = Join-Path $stage "runtime_source_manifest.json"
$developmentDecisionPath = Join-Path $stage "development_decision.json"
$developmentSelectorReceiptPath = Join-Path $stage "development_selector_receipt.json"
$developmentPassFreezePath = Join-Path $stage "development_pass_freeze.json"
$auditDecisionPath = Join-Path $stage "audit50_decision.json"
$auditSelectorReceiptPath = Join-Path $stage "audit50_selector_receipt.json"
$runtimeSourceRoot = Join-Path $frozenPackageRoot "runtime_source"
$runtimeDependencyRoot = $frozenPackageRoot
foreach ($artifact in @(
    @($manifest.acceptance_artifacts.model.uri, $modelPath, $manifest.acceptance_artifacts.model.sha256),
    @($manifest.acceptance_artifacts.training_manifest.uri, $trainingManifestPath, $manifest.acceptance_artifacts.training_manifest.sha256),
    @($manifest.acceptance_artifacts.runtime_freeze.uri, $runtimeFreezePath, $manifest.acceptance_artifacts.runtime_freeze.sha256)
)) {
    Invoke-Gcloud @("storage", "cp", [string]$artifact[0], [string]$artifact[1], "--project", $ProjectId)
    if ((Get-Sha256 ([string]$artifact[1])) -ne [string]$artifact[2]) {
        throw "Downloaded Attempt12 acceptance artifact hash mismatch; staging preserved at $stage"
    }
}
foreach ($artifact in @(
    @((Join-Path $frozenPackageRoot "artifacts/m43_model.pkl"), $modelPath, $manifest.acceptance_artifacts.model.sha256),
    @((Join-Path $frozenPackageRoot "artifacts/training_manifest.json"), $trainingManifestPath, $manifest.acceptance_artifacts.training_manifest.sha256),
    @((Join-Path $frozenPackageRoot "artifacts/runtime_freeze.json"), $runtimeFreezePath, $manifest.acceptance_artifacts.runtime_freeze.sha256),
    @((Join-Path $frozenPackageRoot "artifacts/runtime_source.zip"), $runtimeSourceArchivePath, $manifest.acceptance_artifacts.runtime_source_archive.sha256),
    @((Join-Path $frozenPackageRoot "artifacts/runtime_source_manifest.json"), $runtimeSourceManifestPath, $manifest.acceptance_artifacts.runtime_source_manifest.sha256),
    @((Join-Path $frozenPackageRoot "artifacts/development_decision.json"), $developmentDecisionPath, $manifest.acceptance_artifacts.development_decision.sha256),
    @((Join-Path $frozenPackageRoot "artifacts/development_selector_receipt.json"), $developmentSelectorReceiptPath, $manifest.acceptance_artifacts.development_selector_receipt.sha256),
    @((Join-Path $frozenPackageRoot "artifacts/development_pass_freeze.json"), $developmentPassFreezePath, $manifest.acceptance_artifacts.development_pass_freeze.sha256),
    @((Join-Path $frozenPackageRoot "artifacts/audit50_decision.json"), $auditDecisionPath, $manifest.acceptance_artifacts.audit50_decision.sha256),
    @((Join-Path $frozenPackageRoot "artifacts/audit50_selector_receipt.json"), $auditSelectorReceiptPath, $manifest.acceptance_artifacts.audit50_selector_receipt.sha256)
)) {
    if ((Get-Sha256 ([string]$artifact[0])) -ne [string]$artifact[2]) {
        throw "Attempt12 frozen package/acceptance artifact mismatch; staging preserved at $stage"
    }
    if (-not (Test-Path -LiteralPath ([string]$artifact[1]))) {
        Copy-Item -LiteralPath ([string]$artifact[0]) -Destination ([string]$artifact[1])
    }
    if ((Get-Sha256 ([string]$artifact[1])) -ne [string]$artifact[2]) {
        throw "Attempt12 downloaded artifact differs from frozen package; staging preserved at $stage"
    }
}

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
        throw "Attempt12 population shard DONE/hash chain mismatch: $shard; staging preserved at $stage"
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
$oldPythonPath = $env:PYTHONPATH
$oldDontWriteBytecode = $env:PYTHONDONTWRITEBYTECODE
$env:PYTHONPATH = Join-Path $runtimeSourceRoot "src"
$env:PYTHONDONTWRITEBYTECODE = "1"
try {
    & python -c "import pathlib,sys; from ofc_regular.hu_m43_attempt12_distilled_runtime import validate_distilled_runtime_dependencies,validate_frozen_execution_modules; root=pathlib.Path(sys.argv[1]); validate_frozen_execution_modules(extracted_root=root/'runtime_source',manifest=root/'artifacts/runtime_source_manifest.json',module_names=['ofc_regular.hu_m43_attempt12_distilled_runtime','ofc_regular.evaluate_hu_m4_population','ofc_regular.merge_hu_m4_population_shards']); validate_distilled_runtime_dependencies(root)" $frozenPackageRoot
    if ($LASTEXITCODE -ne 0) { throw "Attempt12 frozen merge execution verification failed; staging preserved at $stage" }
    & python @mergeArgs
    if ($LASTEXITCODE -ne 0) { throw "Attempt12 population shard merger failed ($LASTEXITCODE); staging preserved at $stage" }
}
finally {
    $env:PYTHONPATH = $oldPythonPath
    $env:PYTHONDONTWRITEBYTECODE = $oldDontWriteBytecode
}

$evaluation = Get-Content -LiteralPath $evaluationOutput -Raw | ConvertFrom-Json
$merge = Get-Content -LiteralPath $mergeOutput -Raw | ConvertFrom-Json
$runtime = $evaluation.runtime_config
if ($evaluation.schema -ne "hu_m4_t1_population_evaluation_v1" -or
    [int]$evaluation.paired_seeds_per_opponent -ne [int]$manifest.population_plan.paired_seeds -or
    $runtime.candidate_model_sha256 -ne $manifest.runtime.model_sha256 -or
    $runtime.safety_model_sha256 -ne $manifest.runtime.model_sha256 -or
    $runtime.model_schema -ne "hu_m43_attempt12_t1_second_distilled_selector_v1" -or
    $runtime.artifact_schema -ne "hu_m43_attempt12_t1_second_distilled_pickle_v1" -or
    $runtime.feature_schema -ne "hu_m43_attempt12_lambda_all_legal_public_infoset_features_v1" -or
    $runtime.head_schema -ne "hu_m43_attempt12_policy_delta_safe_tail_heads_v1" -or
    $runtime.model_id -ne $manifest.runtime.model_id -or
    $runtime.action_score_mode -ne "attempt12_lambda_all_legal_distilled_safe_selector_v1" -or
    $runtime.baseline_profile -ne "stage19_p0" -or
    $runtime.runtime_binding_verified -ne $true -or
    $runtime.freeze_manifest_sha256 -ne $manifest.runtime.runtime_freeze_sha256 -or
    $runtime.training_manifest_sha256 -ne $manifest.runtime.training_manifest_sha256 -or
    $runtime.runtime_source_manifest_sha256 -ne $manifest.runtime.runtime_source_manifest_sha256 -or
    $runtime.runtime_source_closure_sha256 -ne $manifest.runtime.runtime_source_closure_sha256 -or
    $runtime.runtime_semantic_closure_sha256 -ne $manifest.runtime.runtime_semantic_closure_sha256 -or
    $runtime.runtime_requirements_sha256 -ne $manifest.runtime.runtime_requirements_sha256 -or
    $runtime.runtime_fingerprint_sha256 -ne $manifest.runtime.runtime_fingerprint_sha256 -or
    $runtime.source_model_manifest_sha256 -ne $manifest.runtime.source_model_manifest_sha256 -or
    $runtime.source_native_manifest_sha256 -ne $manifest.runtime.source_native_manifest_sha256 -or
    $runtime.runtime_dependency_closure_sha256 -ne $manifest.runtime.runtime_dependency_closure_sha256 -or
    $runtime.seed_registry_sha256 -ne $manifest.runtime.seed_registry_sha256 -or
    $runtime.current_profile_used -ne $false -or $runtime.promotion_artifact_contract -ne $true -or
    $merge.status -ne "complete_content_verified" -or
    $merge.population_plan_sha256 -ne $manifest.population_plan.sha256) {
    throw "Merged Attempt12 population violates the frozen distilled runtime/plan contract; staging preserved at $stage"
}

$receipt = [ordered]@{
    schema = $ReceiptSchema
    status = "verified_and_merged"
    run_name = $RunName
    run_manifest_sha256 = $manifestSha
    population_plan_sha256 = $manifest.population_plan.sha256
    model_sha256 = $manifest.runtime.model_sha256
    population_source_archive_sha256 = $manifest.source.sha256
    runtime_source_archive_sha256 = $manifest.runtime.runtime_source_archive_sha256
    runtime_source_manifest_sha256 = $manifest.runtime.runtime_source_manifest_sha256
    runtime_source_closure_sha256 = $manifest.runtime.runtime_source_closure_sha256
    runtime_semantic_closure_sha256 = $manifest.runtime.runtime_semantic_closure_sha256
    source_model_manifest_sha256 = $manifest.runtime.source_model_manifest_sha256
    source_native_manifest_sha256 = $manifest.runtime.source_native_manifest_sha256
    runtime_dependency_closure_sha256 = $manifest.runtime.runtime_dependency_closure_sha256
    seed_registry_sha256 = $SeedRegistrySha256
    development_decision_sha256 = $manifest.launch_preflight.development_decision_sha256
    development_selector_receipt_sha256 = $manifest.launch_preflight.development_selector_receipt_sha256
    development_pass_freeze_sha256 = $manifest.launch_preflight.development_pass_freeze_sha256
    audit50_decision_sha256 = $manifest.launch_preflight.audit50_decision_sha256
    audit50_selector_receipt_sha256 = $manifest.launch_preflight.audit50_selector_receipt_sha256
    development200_full_fit_bound = $true
    audit50_one_shot_go_bound = $true
    model_schema = "hu_m43_attempt12_t1_second_distilled_selector_v1"
    artifact_schema = "hu_m43_attempt12_t1_second_distilled_pickle_v1"
    feature_schema = "hu_m43_attempt12_lambda_all_legal_public_infoset_features_v1"
    head_schema = "hu_m43_attempt12_policy_delta_safe_tail_heads_v1"
    action_score_mode = "attempt12_lambda_all_legal_distilled_safe_selector_v1"
    baseline_profile = "stage19_p0"
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
$receiptPath = Join-Path $stage "receipt.json"
Write-Utf8NoBom $receiptPath (($receipt | ConvertTo-Json -Depth 12) + "`n")

$statusPath = Join-Path $stage "acceptance_status.json"
$acceptArgs = @(
    "-m", "ofc_regular.validate_hu_m43_attempt12_acceptance", "finalize",
    "--population-plan", $planPath,
    "--records", $recordsOutput,
    "--evaluation", $evaluationOutput,
    "--merge-manifest", $mergeOutput,
    "--model", $modelPath,
    "--training-manifest", $trainingManifestPath,
    "--runtime-freeze", $runtimeFreezePath,
    "--runtime-source-archive", $runtimeSourceArchivePath,
    "--runtime-source-manifest", $runtimeSourceManifestPath,
    "--runtime-source-root", $runtimeSourceRoot,
    "--runtime-dependency-root", $runtimeDependencyRoot,
    "--output", $statusPath
)
$oldPythonPath = $env:PYTHONPATH
$oldDontWriteBytecode = $env:PYTHONDONTWRITEBYTECODE
$oldPreference = $ErrorActionPreference
$env:PYTHONPATH = Join-Path $runtimeSourceRoot "src"
$env:PYTHONDONTWRITEBYTECODE = "1"
$ErrorActionPreference = "Continue"
try { $acceptRaw = @(& python @acceptArgs 2>&1); $acceptCode = $LASTEXITCODE }
finally {
    $env:PYTHONPATH = $oldPythonPath
    $env:PYTHONDONTWRITEBYTECODE = $oldDontWriteBytecode
    $ErrorActionPreference = $oldPreference
}
if ($acceptCode -notin @(0, 2)) {
    throw "Attempt12 final acceptance validator failed ($acceptCode):`n$(@($acceptRaw) -join [Environment]::NewLine); staging preserved at $stage"
}
if (-not (Test-Path -LiteralPath $statusPath -PathType Leaf)) {
    throw "Attempt12 acceptance validator did not write its immutable status; staging preserved at $stage"
}
$acceptance = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
$expectedDecision = if ($acceptCode -eq 0) { "complete_go" } else { "complete_no_go" }
if ($acceptance.schema -ne $AcceptanceStatusSchema -or $acceptance.status -ne $expectedDecision -or
    $acceptance.current_profile_mutated -ne $false -or
    $acceptance.runtime_policy_activated -ne $false -or
    $acceptance.full_replacement -ne $false -or
    $acceptance.automatic_activation_authorized -ne $false -or
    ($expectedDecision -eq "complete_go" -and $acceptance.explicit_opt_in_authorized -ne $true) -or
    ($expectedDecision -eq "complete_no_go" -and $acceptance.explicit_opt_in_authorized -ne $false)) {
    throw "Attempt12 acceptance status/exit code mismatch; staging preserved at $stage"
}
$receipt.acceptance_status_sha256 = Get-Sha256 $statusPath
$receipt.acceptance_decision = $expectedDecision
Write-Utf8NoBom $receiptPath (($receipt | ConvertTo-Json -Depth 12) + "`n")
if (Test-Path -LiteralPath $OutputDir) { throw "Attempt12 OutputDir appeared during receive; refusing overwrite. Staging preserved at $stage" }
Move-Item -LiteralPath $stage -Destination $OutputDir
[ordered]@{ receipt = $receipt; acceptance = $acceptance } | ConvertTo-Json -Depth 15
