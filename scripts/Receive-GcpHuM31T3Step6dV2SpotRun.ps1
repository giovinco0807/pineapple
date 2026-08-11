param(
    [Parameter(Mandatory = $true)][string]$RunName,
    [string]$RunDir,
    [string]$OutputDir,
    [string]$ProjectId = "ofc-solver-485418",
    [string]$Bucket = "pokerhu-ofc-solver-485418-training"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $RunDir) { $RunDir = Join-Path $repoRoot "outputs/gcp_runs/$RunName" }
elseif (-not [IO.Path]::IsPathRooted($RunDir)) { $RunDir = Join-Path $repoRoot $RunDir }
$RunDir = [IO.Path]::GetFullPath($RunDir)
if (-not $OutputDir) {
    $OutputDir = Join-Path $repoRoot "outputs/hu_joint_policy/m31_t3_step6d/spot_v2/$RunName"
}
elseif (-not [IO.Path]::IsPathRooted($OutputDir)) { $OutputDir = Join-Path $repoRoot $OutputDir }
$OutputDir = [IO.Path]::GetFullPath($OutputDir)
$oldPythonPath = $env:PYTHONPATH
$oldDontWriteBytecode = $env:PYTHONDONTWRITEBYTECODE
$env:PYTHONPATH = Join-Path $repoRoot "src"
$env:PYTHONDONTWRITEBYTECODE = "1"
try {
    $arguments = @(
        "-m", "ofc_regular.hu_m31_t3_step6d_spot_v2", "receive",
        "--run-dir", $RunDir, "--output-dir", $OutputDir,
        "--project", $ProjectId, "--bucket", $Bucket
    )
    & python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Step 6d v2 receive failed ($LASTEXITCODE)" }
}
finally {
    $env:PYTHONPATH = $oldPythonPath
    $env:PYTHONDONTWRITEBYTECODE = $oldDontWriteBytecode
}
