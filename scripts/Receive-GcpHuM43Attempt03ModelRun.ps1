param(
    [string]$ProjectId = "ofc-solver-485418",
    [string]$Bucket = "pokerhu-ofc-solver-485418-training",
    [Parameter(Mandatory = $true)][string]$RunName,
    [Parameter(Mandatory = $true)][string]$InheritedTrain,
    [Parameter(Mandatory = $true)][string]$FreshTrainFit,
    [Parameter(Mandatory = $true)][string]$ModelFreeze,
    [Parameter(Mandatory = $true)][string]$TrainingFreeze,
    [string]$Manifest,
    [string]$OutputDir
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "HuM43Attempt03ModelSpot.Common.ps1")

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pythonPath = (Get-Command python -ErrorAction Stop | Select-Object -First 1).Source
if (-not $pythonPath) { $pythonPath = "python" }
if ($RunName -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]{2,119}$') { throw "RunName is unsafe" }
$resolvedInherited = Resolve-M43A3Input $InheritedTrain $repoRoot "inherited Attempt02 train200"
$resolvedFresh = Resolve-M43A3Input $FreshTrainFit $repoRoot "fresh Attempt03 train.fit500"
$resolvedModelFreeze = Resolve-M43A3Input $ModelFreeze $repoRoot "pre-row model freeze"
$resolvedTrainingFreeze = Resolve-M43A3Input $TrainingFreeze $repoRoot "training pipeline freeze"

$modelRunsRoot = [IO.Path]::GetFullPath((Join-Path $repoRoot "outputs/hu_joint_policy/m43_attempt03_model_runs"))
if (-not $OutputDir) { $OutputDir = Join-Path $modelRunsRoot $RunName }
$destination = [IO.Path]::GetFullPath((if ([IO.Path]::IsPathRooted($OutputDir)) { $OutputDir } else { Join-Path $repoRoot $OutputDir }))
$allowedPrefix = $modelRunsRoot.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
if (-not $destination.StartsWith($allowedPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "OutputDir must remain under m43_attempt03_model_runs"
}
$receiptPath = Join-Path $destination "receive_receipt.json"

function Assert-ExistingReceipt {
    param([Parameter(Mandatory = $true)]$Receipt)
    if ($Receipt.schema -ne "hu_m43_attempt03_v5_model_receive_receipt_v1" -or
        $Receipt.status -ne "fit_candidate_precalibration_unopened" -or [string]$Receipt.run_name -ne $RunName -or
        (Get-M43A3StrictInteger $Receipt.job_count "receipt.job_count") -ne 30 -or
        (Get-M43A3StrictInteger $Receipt.shard_receipt_count "receipt.shard_receipt_count") -ne 8 -or
        $Receipt.cloud_input_boundary -ne "fit700_only_no_holdouts" -or
        $Receipt.precalibration_opened -ne $false -or $Receipt.sealed_calibration_opened -ne $false -or
        $Receipt.inherited_locked_opened -ne $false -or $Receipt.current_profile_mutated -ne $false -or
        $Receipt.runtime_policy_activated -ne $false -or $Receipt.full_replacement -ne $false) {
        throw "Existing Attempt03 model receive receipt changed"
    }
    $bindings = @{
        "spot_manifest.json" = [string]$Receipt.run_manifest_file_sha256
        "fold_cloud_contract.json" = [string]$Receipt.fold_cloud_contract_file_sha256
        "candidate/candidate_model.pkl" = [string]$Receipt.candidate_model_sha256
        "candidate/fit_bundle.pkl" = [string]$Receipt.fit_bundle_sha256
        "candidate/fit_manifest.json" = [string]$Receipt.fit_manifest_file_sha256
    }
    foreach ($relative in $bindings.Keys) {
        $path = Join-Path $destination $relative
        if (-not (Test-Path -LiteralPath $path -PathType Leaf) -or (Get-M43A3Sha256 $path) -ne $bindings[$relative]) {
            throw "Existing Attempt03 model receive artifact changed: $relative"
        }
    }
}

if (Test-Path -LiteralPath $destination) {
    if (-not (Test-Path -LiteralPath $receiptPath -PathType Leaf)) { throw "Existing Attempt03 model output is partial" }
    $existing = Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json
    Assert-ExistingReceipt $existing
    $existing | ConvertTo-Json -Depth 20
    exit 0
}

New-Item -ItemType Directory -Force -Path $modelRunsRoot | Out-Null
$stageDir = Join-Path $modelRunsRoot (".receiving-{0}-{1}" -f $RunName, [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $stageDir | Out-Null
try {
    $manifestPath = Join-Path $stageDir "spot_manifest.json"
    $prefix = "gs://$Bucket/runs/$RunName"
    $manifestUri = "$prefix/source/m43_attempt03_v5_model_spot_manifest.json"
    if ($Manifest) {
        $resolvedManifest = Resolve-M43A3Input $Manifest $repoRoot "Attempt03 model Spot manifest"
        [IO.File]::WriteAllBytes($manifestPath, [IO.File]::ReadAllBytes($resolvedManifest))
    }
    else { Invoke-M43A3Gcloud @("storage", "cp", $manifestUri, $manifestPath, "--project", $ProjectId) 60 | Out-Null }
    $manifestSha = Get-M43A3Sha256 $manifestPath
    $run = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    Assert-M43A3RunManifest $run $RunName $ProjectId $Bucket
    if ([string]$run.model_freeze_file_sha256 -ne (Get-M43A3Sha256 $resolvedModelFreeze) -or
        [string]$run.training_freeze_file_sha256 -ne (Get-M43A3Sha256 $resolvedTrainingFreeze)) {
        throw "Attempt03 model receive freeze lineage changed"
    }
    if ([string]$run.inputs.fit[0].sha256 -ne (Get-M43A3Sha256 $resolvedInherited) -or
        [string]$run.inputs.fit[1].sha256 -ne (Get-M43A3Sha256 $resolvedFresh)) {
        throw "Attempt03 model receive local fit inputs changed"
    }

    $contractPath = Join-Path $stageDir "fold_cloud_contract.json"
    Invoke-M43A3Gcloud @("storage", "cp", $run.cloud_contract.uri, $contractPath, "--project", $ProjectId) 60 | Out-Null
    if ((Get-M43A3Sha256 $contractPath) -ne [string]$run.cloud_contract.file_sha256) { throw "Downloaded fold contract SHA changed" }
    $validateCode = "from ofc_regular.hu_m43_attempt03_training import load_attempt03_fold_cloud_contract; load_attempt03_fold_cloud_contract(r'$($contractPath.Replace("'", "''"))', repo_root=r'$($repoRoot.Replace("'", "''"))')"
    $validation = Invoke-M43A3ProcessBounded -FilePath $pythonPath -Arguments @("-B", "-c", $validateCode) -TimeoutSeconds 120 `
        -Environment @{ PYTHONPATH = (Join-Path $repoRoot "src"); PYTHONHASHSEED = "0" }
    if ($validation.timed_out -or $validation.exit_code -ne 0) { throw "Downloaded fold contract failed local executable validation: $(@($validation.output) -join ' ')" }

    $foldsDir = Join-Path $stageDir "folds"
    New-Item -ItemType Directory -Path $foldsDir | Out-Null
    $seen = [Collections.Generic.HashSet[int]]::new()
    for ($index = 0; $index -lt 30; $index++) {
        $job = @($run.jobs)[$index]
        $jobDir = Join-Path $foldsDir ("job-{0:D2}" -f $index)
        New-Item -ItemType Directory -Path $jobDir | Out-Null
        foreach ($name in @("estimator.pkl", "job_manifest.json", "DONE.json")) {
            Invoke-M43A3Gcloud @("storage", "cp", "$($job.result_prefix)/$name", (Join-Path $jobDir $name), "--project", $ProjectId) 300 | Out-Null
        }
        $donePath = Join-Path $jobDir "DONE.json"; $jobManifestPath = Join-Path $jobDir "job_manifest.json"; $estimatorPath = Join-Path $jobDir "estimator.pkl"
        $done = Get-Content -LiteralPath $donePath -Raw | ConvertFrom-Json
        Assert-M43A3Done $done $run $index $manifestSha
        if (-not $seen.Add($index)) { throw "Duplicate downloaded DONE job identity: $index" }
        $jobManifest = Get-Content -LiteralPath $jobManifestPath -Raw | ConvertFrom-Json
        if ($jobManifest.schema -ne "hu_m43_attempt03_v5_fold_job_manifest_v1" -or $jobManifest.status -ne "pass" -or
            (Get-M43A3StrictInteger $jobManifest.job_index "job_manifest.job_index") -ne $index -or
            [string]$jobManifest.job_spec_sha256 -ne [string]$job.job_spec_sha256 -or
            [string]$jobManifest.source_sha256 -ne [string]$run.source.sha256 -or
            [string]$jobManifest.run_manifest_sha256 -ne $manifestSha -or
            [string]$jobManifest.cloud_contract_file_sha256 -ne [string]$run.cloud_contract.file_sha256 -or
            [string]$jobManifest.input_bundle_sha256 -ne [string]$run.inputs.input_bundle_sha256 -or
            [string]$jobManifest.training_config_sha256 -ne [string]$run.training_config_sha256 -or
            $jobManifest.fit700_only -ne $true -or (Get-M43A3StrictInteger $jobManifest.holdout_input_count "job_manifest.holdout_input_count") -ne 0 -or
            $jobManifest.current_profile_mutated -ne $false -or $jobManifest.runtime_policy_activated -ne $false -or
            (Get-M43A3Sha256 $estimatorPath) -ne [string]$done.artifact_sha256 -or
            (Get-M43A3Sha256 $jobManifestPath) -ne [string]$done.job_manifest_sha256) {
            throw "Downloaded Attempt03 job hash chain is invalid: $index"
        }
    }
    if ($seen.Count -ne 30) { throw "Attempt03 receive did not cover exact 30 jobs" }

    $shardReceipts = @()
    for ($shard = 0; $shard -lt 8; $shard++) {
        $path = Join-Path $stageDir ("shard-{0:D2}-receipt.json" -f $shard)
        Invoke-M43A3Gcloud @("storage", "cp", "$prefix/results/shard-$($shard.ToString('D2'))/receipt.json", $path, "--project", $ProjectId) 60 | Out-Null
        $receipt = Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
        $expected = @($run.jobs | Where-Object { [int]$_.shard_index -eq $shard } | ForEach-Object { [int]$_.job_index })
        if ($receipt.schema -ne "hu_m43_attempt03_v5_model_spot_shard_receipt_v1" -or $receipt.status -ne "complete" -or
            [string]$receipt.run_name -ne $RunName -or (Get-M43A3StrictInteger $receipt.shard_index "shard receipt") -ne $shard -or
            [string]$receipt.run_manifest_sha256 -ne $manifestSha -or [string]$receipt.source_sha256 -ne [string]$run.source.sha256 -or
            (@($receipt.jobs) -join ",") -ne ($expected -join ",") -or $receipt.current_profile_mutated -ne $false -or
            $receipt.runtime_policy_activated -ne $false) {
            throw "Attempt03 downloaded shard receipt changed: $shard"
        }
        $shardReceipts += [ordered]@{ shard_index = $shard; file_sha256 = Get-M43A3Sha256 $path; jobs = $expected }
    }

    $candidateDir = Join-Path $stageDir "candidate"
    $assemblerArgs = @(
        "-B", "-m", "ofc_regular.assemble_hu_m43_attempt03_model", "assemble-fit",
        "--fold-artifacts-dir", $foldsDir, "--inherited-train", $resolvedInherited,
        "--fresh-train-fit", $resolvedFresh, "--fold-cloud-contract", $contractPath,
        "--model-freeze", $resolvedModelFreeze, "--training-freeze", $resolvedTrainingFreeze,
        "--repo-root", $repoRoot, "--output-dir", $candidateDir, "--run-name", $RunName,
        "--source-sha256", [string]$run.source.sha256, "--run-manifest-sha256", $manifestSha
    )
    $assembly = Invoke-M43A3ProcessBounded -FilePath $pythonPath -Arguments $assemblerArgs -TimeoutSeconds 600 `
        -Environment @{ PYTHONPATH = (Join-Path $repoRoot "src"); PYTHONHASHSEED = "0"; OMP_NUM_THREADS = "1"; OPENBLAS_NUM_THREADS = "1"; MKL_NUM_THREADS = "1"; NUMEXPR_NUM_THREADS = "1" }
    if ($assembly.timed_out -or $assembly.exit_code -ne 0) { throw "Attempt03 frozen fit assembler failed: $(@($assembly.output) -join [Environment]::NewLine)" }
    $candidateModel = Join-Path $candidateDir "candidate_model.pkl"; $fitBundle = Join-Path $candidateDir "fit_bundle.pkl"; $fitManifest = Join-Path $candidateDir "fit_manifest.json"
    if (-not (Test-Path -LiteralPath $candidateModel -PathType Leaf) -or -not (Test-Path -LiteralPath $fitBundle -PathType Leaf) -or
        -not (Test-Path -LiteralPath $fitManifest -PathType Leaf)) { throw "Attempt03 assembler output is incomplete" }
    $fit = Get-Content -LiteralPath $fitManifest -Raw | ConvertFrom-Json
    if ($fit.schema -ne "hu_m43_attempt03_v5_fit_manifest_v1" -or $fit.status -ne "fit_candidate_precalibration_unopened" -or
        [string]$fit.model_sha256 -ne (Get-M43A3Sha256 $candidateModel) -or
        [string]$fit.fit_bundle_sha256 -ne (Get-M43A3Sha256 $fitBundle) -or
        [string]$fit.fold_cloud_contract_sha256 -ne [string]$run.cloud_contract.contract_sha256 -or
        [string]$fit.model_freeze_file_sha256 -ne (Get-M43A3Sha256 $resolvedModelFreeze) -or
        [string]$fit.training_freeze_file_sha256 -ne (Get-M43A3Sha256 $resolvedTrainingFreeze) -or
        $fit.precalibration.opened -ne $false -or $fit.sealed_calibration.opened -ne $false -or
        $fit.inherited_locked.opened -ne $false -or $fit.current_profile_mutated -ne $false -or
        $fit.runtime_policy_activated -ne $false -or $fit.full_replacement -ne $false) {
        throw "Attempt03 assembled fit candidate lineage changed"
    }

    $receive = [ordered]@{
        schema = "hu_m43_attempt03_v5_model_receive_receipt_v1"; status = "fit_candidate_precalibration_unopened"
        run_name = $RunName; job_count = 30; shard_receipt_count = 8
        run_manifest_file_sha256 = $manifestSha; source_sha256 = [string]$run.source.sha256
        fold_cloud_contract_file_sha256 = Get-M43A3Sha256 $contractPath
        fold_cloud_contract_sha256 = [string]$run.cloud_contract.contract_sha256
        input_bundle_sha256 = [string]$run.inputs.input_bundle_sha256; training_config_sha256 = [string]$run.training_config_sha256
        model_freeze_file_sha256 = Get-M43A3Sha256 $resolvedModelFreeze
        training_freeze_file_sha256 = Get-M43A3Sha256 $resolvedTrainingFreeze
        candidate_model_sha256 = Get-M43A3Sha256 $candidateModel; fit_bundle_sha256 = Get-M43A3Sha256 $fitBundle
        fit_manifest_file_sha256 = Get-M43A3Sha256 $fitManifest; shard_receipts = $shardReceipts
        cloud_input_boundary = "fit700_only_no_holdouts"; precalibration_opened = $false
        sealed_calibration_opened = $false; inherited_locked_opened = $false
        current_profile_mutated = $false; runtime_policy_activated = $false; full_replacement = $false
    }
    Write-M43A3Utf8CreateNew (Join-Path $stageDir "receive_receipt.json") (($receive | ConvertTo-Json -Depth 20) + "`n")
    Move-Item -LiteralPath $stageDir -Destination $destination
    $receive | ConvertTo-Json -Depth 20
}
catch {
    if ($stageDir.StartsWith($allowedPrefix, [StringComparison]::OrdinalIgnoreCase) -and (Test-Path -LiteralPath $stageDir)) {
        Remove-Item -LiteralPath $stageDir -Recurse -Force -ErrorAction SilentlyContinue
    }
    throw
}
