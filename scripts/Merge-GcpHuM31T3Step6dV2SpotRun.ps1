param(
    [Parameter(Mandatory = $true)][string]$RunName,
    [string]$ReceiveDir,
    [string]$MergeDir,
    [ValidateSet("auto", "candidate01_tail_diagnostic", "candidate02_tail_diagnostic", "full_performance_development")]
    [string]$Scope = "auto"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $ReceiveDir) {
    $ReceiveDir = Join-Path $repoRoot "outputs/hu_joint_policy/m31_t3_step6d/spot_v2/$RunName"
}
elseif (-not [IO.Path]::IsPathRooted($ReceiveDir)) {
    $ReceiveDir = Join-Path $repoRoot $ReceiveDir
}
$ReceiveDir = [IO.Path]::GetFullPath($ReceiveDir)
if (-not (Test-Path -LiteralPath $ReceiveDir -PathType Container)) {
    throw "Step 6d v2 receive directory is missing: $ReceiveDir"
}
if (-not $MergeDir) {
    $MergeDir = Join-Path $repoRoot "outputs/hu_joint_policy/m31_t3_step6d/spot_v2_merge/$RunName"
}
elseif (-not [IO.Path]::IsPathRooted($MergeDir)) {
    $MergeDir = Join-Path $repoRoot $MergeDir
}
$MergeDir = [IO.Path]::GetFullPath($MergeDir)

$oldPythonPath = $env:PYTHONPATH
$oldDontWriteBytecode = $env:PYTHONDONTWRITEBYTECODE
$env:PYTHONPATH = Join-Path $repoRoot "src"
$env:PYTHONDONTWRITEBYTECODE = "1"
try {
    $arguments = @(
        "-m", "ofc_regular.merge_hu_m31_t3_step6d_spot_v2",
        "--receive-dir", $ReceiveDir,
        "--summary-output", (Join-Path $MergeDir "summary.json"),
        "--validation-output", (Join-Path $MergeDir "validation.json"),
        "--expected-run-name", $RunName,
        "--scope", $Scope
    )
    & python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Step 6d v2 receipt merge failed ($LASTEXITCODE)" }
}
finally {
    $env:PYTHONPATH = $oldPythonPath
    $env:PYTHONDONTWRITEBYTECODE = $oldDontWriteBytecode
}
