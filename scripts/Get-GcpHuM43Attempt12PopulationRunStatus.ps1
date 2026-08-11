param(
    [Parameter(Mandatory = $true)][string]$RunName,
    [string]$ProjectId = "ofc-solver-485418",
    [string]$Bucket = "pokerhu-ofc-solver-485418-training"
)

$ErrorActionPreference = "Stop"
$ManifestSchema = "hu_m43_attempt12_population_spot_manifest_v1"
$DoneSchema = "hu_m43_attempt12_population_spot_done_v1"
$StatusSchema = "hu_m43_attempt12_population_spot_status_v1"
$PopulationPlanSha256 = "66d9c3dc50bcd6ae2ce72e2a558dcfae5720d3cb2387a16ef810d1498f4db462"
$SeedRegistrySha256 = "de63af1ded1bfe6ff7c92dc6bcc8aedec9c408956e687c856198143bffca8255"
if ($RunName -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$') { throw "RunName contains unsafe characters" }

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}
function Convert-RequiredJsonInteger($Value, [string]$Label) {
    if ($null -eq $Value -or $Value -is [bool] -or $Value -is [string] -or
        $Value -is [single] -or $Value -is [double] -or $Value -is [decimal]) {
        throw "$Label must be a JSON integer"
    }
    try { return [long]$Value } catch { throw "$Label must be a JSON integer" }
}
function Invoke-Gcloud([string[]]$Arguments) {
    $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    try { $output = & gcloud @Arguments 2>&1; $code = $LASTEXITCODE }
    finally { $ErrorActionPreference = $old }
    if ($code -ne 0) { throw "gcloud failed ($code): $(@($output) -join [Environment]::NewLine)" }
    return @($output)
}
function Try-GetGcsJson([string]$Uri) {
    for ($attempt = 1; $attempt -le 2; $attempt++) {
        $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
        try { $output = & gcloud storage cat $Uri --project $ProjectId 2>&1; $code = $LASTEXITCODE }
        finally { $ErrorActionPreference = $old }
        if ($code -eq 0) {
            try { return ((@($output) -join "`n") | ConvertFrom-Json) }
            catch { throw "Invalid JSON in GCS status object: $Uri" }
        }
        $message = @($output) -join "`n"
        if ($message -notmatch '(?i)not found|does not exist|No URLs matched|404') {
            throw "Unable to read Attempt12 population status object: $Uri`n$message"
        }
        if ($attempt -lt 2) { Start-Sleep -Milliseconds 250 }
    }
    return $null
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$manifestPath = Join-Path $repoRoot "outputs/gcp_runs/$RunName/population_run_manifest.json"
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) { throw "Frozen Attempt12 population manifest is missing: $manifestPath" }
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$manifestSha = Get-Sha256 $manifestPath
if ($manifest.schema -ne $ManifestSchema -or $manifest.run_name -ne $RunName -or
    $manifest.project_id -ne $ProjectId -or $manifest.bucket -ne $Bucket -or
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
    $manifest.acceptance_artifacts.development_decision.sha256 -ne $manifest.launch_preflight.development_decision_sha256 -or
    $manifest.acceptance_artifacts.development_selector_receipt.sha256 -ne $manifest.launch_preflight.development_selector_receipt_sha256 -or
    $manifest.acceptance_artifacts.development_pass_freeze.sha256 -ne $manifest.launch_preflight.development_pass_freeze_sha256 -or
    $manifest.acceptance_artifacts.audit50_decision.sha256 -ne $manifest.launch_preflight.audit50_decision_sha256 -or
    $manifest.acceptance_artifacts.audit50_selector_receipt.sha256 -ne $manifest.launch_preflight.audit50_selector_receipt_sha256 -or
    $manifest.checkpoint.unit -ne "completed_shard" -or
    $manifest.checkpoint.resume_missing_shards_only -ne $true -or
    $manifest.checkpoint.done_commit_last -ne $true -or
    $manifest.heartbeat.schema -ne $StatusSchema -or
    $manifest.heartbeat.required -ne $true -or
    $manifest.source_boundary.provenance_json_evidence_packaged -ne $true -or
    $manifest.no_runtime_activation -ne $true -or $manifest.current_profile_mutated -ne $false) {
    throw "Frozen Attempt12 population manifest identity mismatch"
}
$shardCount = Convert-RequiredJsonInteger $manifest.shards.count "manifest.shards.count"
if ($shardCount -ne 20 -or [int]$manifest.population_plan.paired_seeds -ne 1000 -or
    [long]$manifest.population_plan.seed -ne 220108071901 -or
    [long]$manifest.population_plan.seed_stride -ne 1000003 -or
    [int]$manifest.population_plan.paired_seeds_per_shard -ne 50) {
    throw "Frozen Attempt12 population manifest is not the fixed 20x50 schedule"
}
$prefix = "gs://$Bucket/runs/$RunName"
$old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
try { $rawDone = & gcloud storage ls "$prefix/results/*/DONE" --project $ProjectId 2>&1; $doneCode = $LASTEXITCODE }
finally { $ErrorActionPreference = $old }
if ($doneCode -eq 0) { $doneUris = @($rawDone | Where-Object { $_ -match '/DONE$' }) }
elseif ((@($rawDone) -join "`n") -match '(?i)not found|does not exist|No URLs matched|404') { $doneUris = @() }
else { throw "Unable to list Attempt12 population DONE objects: $(@($rawDone) -join [Environment]::NewLine)" }

$doneRows = @()
$seen = [System.Collections.Generic.HashSet[int]]::new()
foreach ($uri in $doneUris) {
    if ($uri -notmatch '/results/shard-(\d{4})/DONE$') { throw "Attempt12 DONE URI does not encode a canonical shard: $uri" }
    $uriShard = [int]$Matches[1]
    $expectedUri = "$prefix/results/shard-{0:D4}/DONE" -f $uriShard
    if ([string]$uri -cne $expectedUri) { throw "Attempt12 DONE URI is outside the frozen run prefix: $uri" }
    if ($uriShard -lt 0 -or $uriShard -ge $shardCount) { throw "Attempt12 DONE shard is outside frozen range: $uriShard" }
    if (-not $seen.Add($uriShard)) { throw "Duplicate Attempt12 DONE shard: $uriShard" }
    $row = ((Invoke-Gcloud @("storage", "cat", $uri, "--project", $ProjectId)) -join "`n") | ConvertFrom-Json
    $rowShard = Convert-RequiredJsonInteger $row.shard "DONE.shard"
    if ($row.schema -ne $DoneSchema -or $row.status -ne "complete" -or
        $row.run_name -ne $RunName -or $rowShard -ne $uriShard -or
        $row.manifest_sha256 -ne $manifestSha -or $row.source_sha256 -ne $manifest.source.sha256 -or
        $row.model_sha256 -ne $manifest.runtime.model_sha256 -or
        $row.evaluation_sha256 -notmatch '^[0-9a-f]{64}$' -or
        $row.records_sha256 -notmatch '^[0-9a-f]{64}$' -or
        $row.current_profile_mutated -ne $false -or $row.no_runtime_activation -ne $true) {
        throw "Invalid Attempt12 population DONE object: $uri"
    }
    $doneRows += $row
}

$statusRows = @()
if ($doneRows.Count -lt $shardCount) {
    $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    try { $rawStatus = & gcloud storage ls "$prefix/status/*.json" --project $ProjectId 2>&1; $statusCode = $LASTEXITCODE }
    finally { $ErrorActionPreference = $old }
    if ($statusCode -eq 0) {
        $seenStatus = [System.Collections.Generic.HashSet[int]]::new()
        foreach ($uri in @($rawStatus | Where-Object { $_ -match '\.json$' })) {
            if ($uri -notmatch '/status/shard-(\d+)\.json$') { throw "Noncanonical Attempt12 population status URI: $uri" }
            $uriShard = [int]$Matches[1]
            $expectedUri = "$prefix/status/shard-$uriShard.json"
            if ([string]$uri -cne $expectedUri -or $uriShard -lt 0 -or $uriShard -ge $shardCount) {
                throw "Attempt12 population status URI is outside the frozen run: $uri"
            }
            if (-not $seenStatus.Add($uriShard)) { throw "Duplicate Attempt12 population status shard: $uriShard" }
            if ($seen.Contains($uriShard)) { continue }
            $row = Try-GetGcsJson $uri
            if ($null -eq $row) { continue }
            $rowShard = Convert-RequiredJsonInteger $row.shard "status.shard"
            if ($row.schema -ne $StatusSchema -or $row.run_name -ne $RunName -or
                $rowShard -ne $uriShard -or $row.manifest_sha256 -ne $manifestSha -or
                [string]$row.state -notin @("booting", "evaluating", "failed", "complete")) {
                throw "Invalid or stale Attempt12 population status object: $uri"
            }
            $statusRows += $row
        }
    }
    elseif ((@($rawStatus) -join "`n") -notmatch '(?i)not found|does not exist|No URLs matched|404') {
        throw "Unable to list Attempt12 population status objects: $(@($rawStatus) -join [Environment]::NewLine)"
    }
}

$vmPrefix = (($RunName.ToLowerInvariant() -replace '[^a-z0-9-]', '-').Trim('-'))
if ($vmPrefix -notmatch '^[a-z]') { $vmPrefix = "m43a8p-$vmPrefix" }
if ($vmPrefix.Length -gt 56) { $vmPrefix = $vmPrefix.Substring(0, 56).TrimEnd('-') }
$instancesRaw = Invoke-Gcloud @("compute", "instances", "list", "--project", $ProjectId, "--filter", "name~'^$vmPrefix-s'", "--format=json")
$instances = if ($instancesRaw) { @((@($instancesRaw) -join "`n") | ConvertFrom-Json) } else { @() }
$completed = @($doneRows | ForEach-Object { [int]$_.shard } | Sort-Object)
[ordered]@{
    schema = "hu_m43_attempt12_population_spot_status_report_v1"
    run_name = $RunName
    state = $(if ($doneRows.Count -eq $shardCount) { "complete" } elseif (@($statusRows | Where-Object { $_.state -eq "failed" }).Count) { "failed" } else { "running" })
    manifest_sha256 = $manifestSha
    shards_total = $shardCount
    shards_complete = $doneRows.Count
    shards_failed = @($statusRows | Where-Object { $_.state -eq "failed" }).Count
    shards_running = @($statusRows | Where-Object { $_.state -eq "evaluating" }).Count
    completed_indices = $completed
    missing_indices = @(0..($shardCount - 1) | Where-Object { $_ -notin $completed })
    active_instances = @($instances).Count
    instances = @($instances | ForEach-Object { [ordered]@{ name = $_.name; zone = ([string]$_.zone).Split('/')[-1]; status = $_.status; machine_type = ([string]$_.machineType).Split('/')[-1] } })
    model_schema = "hu_m43_attempt12_t1_second_distilled_selector_v1"
    action_score_mode = "attempt12_lambda_all_legal_distilled_safe_selector_v1"
    baseline_profile = "stage19_p0"
    population_plan_sha256 = $PopulationPlanSha256
    seed_registry_sha256 = $SeedRegistrySha256
    development200_full_fit_bound = $true
    audit50_one_shot_go_bound = $true
    no_runtime_activation = $true
    current_profile_mutated = $false
} | ConvertTo-Json -Depth 10
