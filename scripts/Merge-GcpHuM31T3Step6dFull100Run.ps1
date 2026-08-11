param(
    [Parameter(Mandatory = $true)][string]$RunName,
    [string]$ReceiveDir,
    [string]$MergeDir,
    [string]$PlanPath
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

if (-not $ReceiveDir) {
    $ReceiveDir = Join-Path (
        $repoRoot
    ) "outputs/hu_joint_policy/m31_t3_step6d/full100_spot/$RunName"
}
elseif (-not [IO.Path]::IsPathRooted($ReceiveDir)) {
    $ReceiveDir = Join-Path $repoRoot $ReceiveDir
}
$ReceiveDir = [IO.Path]::GetFullPath($ReceiveDir)
if (-not (Test-Path -LiteralPath $ReceiveDir -PathType Container)) {
    throw "Full100 receive directory is missing: $ReceiveDir"
}

if (-not $MergeDir) {
    $MergeDir = Join-Path (
        $repoRoot
    ) "outputs/hu_joint_policy/m31_t3_step6d/full100_merge/$RunName"
}
elseif (-not [IO.Path]::IsPathRooted($MergeDir)) {
    $MergeDir = Join-Path $repoRoot $MergeDir
}
$MergeDir = [IO.Path]::GetFullPath($MergeDir)

if (-not $PlanPath) {
    $PlanPath = Join-Path (
        $repoRoot
    ) "outputs/hu_joint_policy/m31_t3_step6d/candidate02_development/full100_plan_v1.json"
}
elseif (-not [IO.Path]::IsPathRooted($PlanPath)) {
    $PlanPath = Join-Path $repoRoot $PlanPath
}
$PlanPath = [IO.Path]::GetFullPath($PlanPath)
if (-not (Test-Path -LiteralPath $PlanPath -PathType Leaf)) {
    throw "Full100 frozen plan is missing: $PlanPath"
}

$oldPythonPath = $env:PYTHONPATH
$oldDontWriteBytecode = $env:PYTHONDONTWRITEBYTECODE
$env:PYTHONPATH = Join-Path $repoRoot "src"
$env:PYTHONDONTWRITEBYTECODE = "1"
try {
    & python -m ofc_regular.merge_hu_m31_t3_step6d_full100_received_v1 `
        --receive-dir $ReceiveDir `
        --expected-run-name $RunName `
        --plan $PlanPath `
        --summary-output (Join-Path $MergeDir "summary.json") `
        --validation-output (Join-Path $MergeDir "validation.json")
    if ($LASTEXITCODE -ne 0) {
        throw "Full100 receipt-bound scientific merge failed ($LASTEXITCODE)"
    }
}
finally {
    $env:PYTHONPATH = $oldPythonPath
    $env:PYTHONDONTWRITEBYTECODE = $oldDontWriteBytecode
}
