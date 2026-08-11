param(
    [Parameter(Mandatory = $true)][string]$RunName,
    [string]$ProjectId = "ofc-solver-485418",
    [string]$Bucket = "pokerhu-ofc-solver-485418-training",
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"

if ($RunName -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$') {
    throw "RunName may only contain letters, numbers, dot, underscore, and dash"
}

function Get-Sha256 {
    param([string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Invoke-Gcloud {
    param([string[]]$Arguments)
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & gcloud @Arguments 2>&1
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($exitCode -ne 0) {
        throw "gcloud failed ($exitCode): gcloud $($Arguments -join ' ')`n$(@($output) -join [Environment]::NewLine)"
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runDir = Join-Path $repoRoot "outputs/gcp_runs/$RunName"
$runManifestPath = Join-Path $runDir "model_run_manifest.json"
if (-not (Test-Path -LiteralPath $runManifestPath -PathType Leaf)) {
    throw "Local frozen M4.2 model manifest is missing: $runManifestPath"
}
$runManifest = Get-Content -LiteralPath $runManifestPath -Raw | ConvertFrom-Json
$runManifestSha = Get-Sha256 $runManifestPath
if ($runManifest.schema -ne "hu_m42_model_run_manifest_v1" -or
    $runManifest.run_name -ne $RunName -or
    $runManifest.project_id -ne $ProjectId -or
    $runManifest.bucket -ne $Bucket -or
    $runManifest.no_runtime_activation -ne $true) {
    throw "Local run manifest identity or no-activation boundary is invalid"
}
if ($runManifest.teacher_run_name -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$') {
    throw "Frozen teacher run name is not path-safe"
}

$modelRunsRoot = [System.IO.Path]::GetFullPath((Join-Path $repoRoot ("outputs/hu_joint_policy/m42_spot/{0}/model_runs" -f $runManifest.teacher_run_name)))
if ($OutputDir) {
    if (-not [System.IO.Path]::IsPathRooted($OutputDir)) {
        $OutputDir = Join-Path $repoRoot $OutputDir
    }
    $OutputDir = [System.IO.Path]::GetFullPath($OutputDir)
}
else {
    $OutputDir = [System.IO.Path]::GetFullPath((Join-Path $modelRunsRoot $RunName))
}
$rootPrefix = $modelRunsRoot.TrimEnd([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar
if (-not $OutputDir.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "OutputDir must be a child of the frozen M4.2 model_runs directory; refusing a possible profile/current overwrite"
}

$prefix = "gs://$Bucket/runs/$RunName/results"
$objects = [ordered]@{
    DONE = "DONE"
    model = "model.pkl"
    training_manifest = "training_manifest.json"
    validation = "validation.json"
    training_stdout = "training_stdout.json"
    startup_log = "startup.log"
    status = "status.json"
}

if (Test-Path -LiteralPath $OutputDir) {
    if (-not (Test-Path -LiteralPath $OutputDir -PathType Container)) {
        throw "OutputDir exists and is not a directory: $OutputDir"
    }
    $existingReceiptPath = Join-Path $OutputDir "receipt.json"
    $existingDonePath = Join-Path $OutputDir "DONE"
    $requiredExisting = @($objects.Values | ForEach-Object { Join-Path $OutputDir $_ }) + @($existingReceiptPath)
    if (@($requiredExisting | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) }).Count -ne 0) {
        throw "OutputDir already exists but is not a complete verified receipt; refusing to overwrite it: $OutputDir"
    }
    $existingReceipt = Get-Content -LiteralPath $existingReceiptPath -Raw | ConvertFrom-Json
    $existingDone = Get-Content -LiteralPath $existingDonePath -Raw | ConvertFrom-Json
    if ($existingReceipt.schema -ne "hu_m42_model_receipt_v1" -or
        $existingReceipt.status -ne "verified" -or
        $existingReceipt.run_name -ne $RunName -or
        $existingReceipt.run_manifest_sha256 -ne $runManifestSha -or
        $existingReceipt.source_sha256 -ne $existingDone.source_sha256 -or
        $existingReceipt.model_sha256 -ne $existingDone.model_sha256 -or
        $existingReceipt.training_manifest_sha256 -ne $existingDone.training_manifest_sha256 -or
        $existingReceipt.validation_sha256 -ne $existingDone.validation_sha256 -or
        $existingReceipt.no_runtime_activation -ne $true -or
        $existingDone.schema -ne "hu_m42_model_done_v1" -or
        $existingDone.status -ne "complete" -or
        $existingDone.run_name -ne $RunName -or
        $existingDone.run_manifest_sha256 -ne $runManifestSha -or
        $existingDone.source_sha256 -ne $runManifest.source.sha256 -or
        $existingDone.no_runtime_activation -ne $true -or
        (Get-Sha256 (Join-Path $OutputDir "model.pkl")) -ne $existingDone.model_sha256 -or
        (Get-Sha256 (Join-Path $OutputDir "training_manifest.json")) -ne $existingDone.training_manifest_sha256 -or
        (Get-Sha256 (Join-Path $OutputDir "validation.json")) -ne $existingDone.validation_sha256 -or
        (Get-Sha256 (Join-Path $OutputDir "training_stdout.json")) -ne $existingDone.training_stdout_sha256) {
        throw "Existing M4.2 model receipt is stale or has a broken hash chain; refusing to overwrite it"
    }
    $existingReceipt | ConvertTo-Json -Depth 20
    exit 0
}

New-Item -ItemType Directory -Path $modelRunsRoot -Force | Out-Null
$stageDir = Join-Path $modelRunsRoot (".receiving-{0}-{1}" -f $RunName, [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $stageDir | Out-Null
foreach ($name in $objects.Keys) {
    Invoke-Gcloud @("storage", "cp", "$prefix/$($objects[$name])", (Join-Path $stageDir $objects[$name]), "--project", $ProjectId)
}

$donePath = Join-Path $stageDir "DONE"
$modelPath = Join-Path $stageDir "model.pkl"
$trainingManifestPath = Join-Path $stageDir "training_manifest.json"
$validationPath = Join-Path $stageDir "validation.json"
$stdoutPath = Join-Path $stageDir "training_stdout.json"
$done = Get-Content -LiteralPath $donePath -Raw | ConvertFrom-Json
$trainingManifest = Get-Content -LiteralPath $trainingManifestPath -Raw | ConvertFrom-Json
$validation = Get-Content -LiteralPath $validationPath -Raw | ConvertFrom-Json

if ($done.schema -ne "hu_m42_model_done_v1" -or
    $done.status -ne "complete" -or
    $done.run_name -ne $RunName -or
    $done.run_manifest_sha256 -ne $runManifestSha -or
    $done.source_sha256 -ne $runManifest.source.sha256 -or
    $done.no_runtime_activation -ne $true) {
    throw "Invalid or stale M4.2 model DONE object"
}
if ((Get-Sha256 $modelPath) -ne $done.model_sha256 -or
    (Get-Sha256 $trainingManifestPath) -ne $done.training_manifest_sha256 -or
    (Get-Sha256 $validationPath) -ne $done.validation_sha256 -or
    (Get-Sha256 $stdoutPath) -ne $done.training_stdout_sha256) {
    throw "Downloaded M4.2 model output hash chain is invalid"
}
if ($validation.schema -ne "hu_m42_model_validation_v1" -or
    $validation.status -ne "pass" -or
    $validation.run_name -ne $RunName -or
    $validation.run_manifest_sha256 -ne $runManifestSha -or
    $validation.model_sha256 -ne $done.model_sha256 -or
    $validation.no_runtime_activation -ne $true) {
    throw "M4.2 worker validation is missing or invalid"
}
if ($trainingManifest.schema -ne "hu_m4_t1_joint_training_manifest_v2" -or
    $trainingManifest.training_config.action_score_mode -ne "negative_regret_ranker_v2" -or
    [int]$trainingManifest.training_config.cross_fit_folds -ne [int]$runManifest.training_config.cross_fit_folds -or
    [int]$trainingManifest.training_config.iterations -ne [int]$runManifest.training_config.iterations -or
    $trainingManifest.cross_fit.status -ne "pass" -or
    [int]$trainingManifest.cross_fit.fold_assignment.identity_leakage_count -ne 0 -or
    [int]$trainingManifest.cross_fit.predictor_lineage.identity_leakage_count -ne 0 -or
    $trainingManifest.cross_fit.predictor_lineage.all_outer_validation_identities_excluded -ne $true -or
    [int]$trainingManifest.split_integrity.seed_overlap_count -ne 0 -or
    $trainingManifest.hidden_discard_safety.status -ne "pass" -or
    $trainingManifest.calibration_partition.status -ne "pass" -or
    $trainingManifest.locked_holdout_used_for_threshold_or_training -ne $false -or
    $trainingManifest.teacher_value_runtime_gate -ne $false -or
    $trainingManifest.runtime_lock.candidate_model_sha256 -ne $done.model_sha256 -or
    $trainingManifest.runtime_lock.safety_model_sha256 -ne $done.model_sha256) {
    throw "Downloaded M4.2 training manifest violates the frozen no-leak contract"
}
foreach ($split in @("train", "calibration", "locked_holdout")) {
    $expected = $runManifest.inputs.$split
    $actual = @($trainingManifest.inputs.$split)
    if ($actual.Count -ne 1 -or
        [int]$actual[0].rows -ne [int]$expected.rows -or
        [string]$actual[0].sha256 -ne [string]$expected.sha256) {
        throw "M4.2 training input provenance mismatch for split $split"
    }
}

$receipt = [ordered]@{
    schema = "hu_m42_model_receipt_v1"
    status = "verified"
    run_name = $RunName
    teacher_run_name = $runManifest.teacher_run_name
    run_manifest_sha256 = $runManifestSha
    source_sha256 = $done.source_sha256
    model_path = [System.IO.Path]::GetFullPath((Join-Path $OutputDir "model.pkl"))
    model_sha256 = $done.model_sha256
    training_manifest_path = [System.IO.Path]::GetFullPath((Join-Path $OutputDir "training_manifest.json"))
    training_manifest_sha256 = $done.training_manifest_sha256
    validation_sha256 = $done.validation_sha256
    promotion_status = $trainingManifest.promotion_status
    selected_threshold = $trainingManifest.calibration.selected_threshold
    safety_estimator_class = $validation.safety_estimator_class
    calibration = $trainingManifest.calibration
    locked_holdout = $trainingManifest.locked_holdout
    teacher_value_status = $trainingManifest.teacher_value_status
    teacher_metrics_are_realized_match_ev = $false
    locked_holdout_used_for_threshold_search = $false
    current_profile_mutated = $false
    no_runtime_activation = $true
    received_at = (Get-Date).ToUniversalTime().ToString("o")
}
$receiptPath = Join-Path $stageDir "receipt.json"
[System.IO.File]::WriteAllText(
    $receiptPath,
    (($receipt | ConvertTo-Json -Depth 20) + "`n"),
    [System.Text.UTF8Encoding]::new($false)
)
if (Test-Path -LiteralPath $OutputDir) {
    throw "OutputDir appeared during receive; refusing to overwrite it. Verified staging is preserved at $stageDir"
}
Move-Item -LiteralPath $stageDir -Destination $OutputDir
$receipt | ConvertTo-Json -Depth 20
