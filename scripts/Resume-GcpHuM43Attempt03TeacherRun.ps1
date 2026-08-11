param(
    [Parameter(Mandatory = $true)][switch]$ResumeClaimedPrecalFinalize,
    [string]$ProjectId = "ofc-solver-485418",
    [string]$Bucket = "pokerhu-ofc-solver-485418-training",
    [Parameter(Mandatory = $true)][string]$RunName,
    [Parameter(Mandatory = $true)][string]$ExpectedStageName,
    [Parameter(Mandatory = $true)][string]$ExpectedOpenClaimFileSha256,
    [Parameter(Mandatory = $true)][string]$ExpectedDataContractFileSha256,
    [Parameter(Mandatory = $true)][string]$ExpectedMergedPrecalFileSha256,
    [switch]$AuditOnly
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if (-not $ResumeClaimedPrecalFinalize) {
    throw "ResumeClaimedPrecalFinalize must be explicitly selected"
}
if ($RunName -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$') {
    throw "RunName is not path-safe"
}
if ($ExpectedStageName -notmatch '^\.recv-[0-9a-f]{32}$') {
    throw "ExpectedStageName must name one exact Attempt03 receive stage"
}
foreach ($digest in @(
    $ExpectedOpenClaimFileSha256,
    $ExpectedDataContractFileSha256,
    $ExpectedMergedPrecalFileSha256
)) {
    if ($digest -cnotmatch '^[0-9a-f]{64}$') {
        throw "Every expected recovery digest must be lowercase SHA-256"
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runDir = Join-Path $repoRoot "outputs/gcp_runs/$RunName"
$stageDir = Join-Path $repoRoot "outputs/hu_joint_policy/m43_attempt03_precal/$ExpectedStageName"
$finalOutputDir = Join-Path $repoRoot "outputs/hu_joint_policy/m43_attempt03_precal/$RunName"
$claimPath = Join-Path $repoRoot "outputs/hu_joint_policy/m43_attempt03_precal_open_claims/$RunName/M43_ATTEMPT03_PRECAL_OPEN_CLAIM.json"
$fitReceiptPath = Join-Path $repoRoot "outputs/hu_joint_policy/m43_attempt03_teacher/$RunName/fit_receipt.json"
$arguments = @(
    "-B", "-m", "ofc_regular.recover_hu_m43_attempt03_precal_finalize",
    $(if ($AuditOnly) { "audit" } else { "finalize" }),
    "--repo-root", $repoRoot,
    "--run-name", $RunName,
    "--project-id", $ProjectId,
    "--bucket", $Bucket,
    "--plan", (Join-Path $repoRoot "configs/hu_joint_policy_m43_attempt03.json"),
    "--preflight-receipt", (Join-Path $runDir "m43_attempt03_preflight_receipt.json"),
    "--claim", $claimPath,
    "--rp-dir", (Join-Path $runDir "rp"),
    "--stage-dir", $stageDir,
    "--final-output-dir", $finalOutputDir,
    "--fit-receipt", $fitReceiptPath,
    "--model-freeze", (Join-Path $repoRoot "configs/hu_joint_policy_m43_attempt03_model_freeze.json"),
    "--original-freeze", (Join-Path $repoRoot "configs/hu_joint_policy_m43_attempt03_model_freeze_original_ce47.json"),
    "--freeze-lineage", (Join-Path $repoRoot "configs/hu_joint_policy_m43_attempt03_model_freeze_lineage.json"),
    "--expected-claim-sha256", $ExpectedOpenClaimFileSha256,
    "--expected-contract-file-sha256", $ExpectedDataContractFileSha256,
    "--expected-merged-sha256", $ExpectedMergedPrecalFileSha256
)

$oldPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = Join-Path $repoRoot "src"
try {
    $output = @(& python @arguments 2>&1)
    if ($LASTEXITCODE -ne 0) {
        throw "Attempt03 claimed pre-cal finalize recovery failed: $($output -join "`n")"
    }
    $output -join "`n"
}
finally {
    $env:PYTHONPATH = $oldPythonPath
}
