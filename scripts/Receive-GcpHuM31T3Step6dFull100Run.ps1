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
if (-not $RunDir) {
    $RunDir = Join-Path $repoRoot "outputs/gcp_runs/$RunName"
}
elseif (-not [IO.Path]::IsPathRooted($RunDir)) {
    $RunDir = Join-Path $repoRoot $RunDir
}
if (-not $OutputDir) {
    $OutputDir = Join-Path (
        $repoRoot
    ) "outputs/hu_joint_policy/m31_t3_step6d/full100_spot/$RunName"
}
elseif (-not [IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir = Join-Path $repoRoot $OutputDir
}
if (
    $ProjectId -cne "ofc-solver-485418" -or
    $Bucket -cne "pokerhu-ofc-solver-485418-training"
) {
    throw "Full100 receive target differs from the frozen authorization"
}
$oldPythonPath = $env:PYTHONPATH
$oldDontWriteBytecode = $env:PYTHONDONTWRITEBYTECODE
$env:PYTHONPATH = Join-Path $repoRoot "src"
$env:PYTHONDONTWRITEBYTECODE = "1"
try {
    & python -m ofc_regular.hu_m31_t3_step6d_full100_spot_v1 receive `
        --run-dir ([IO.Path]::GetFullPath($RunDir)) `
        --output-dir ([IO.Path]::GetFullPath($OutputDir)) `
        --project $ProjectId --bucket $Bucket
    if ($LASTEXITCODE -ne 0) {
        throw "Full100 receive command failed ($LASTEXITCODE)"
    }
}
finally {
    $env:PYTHONPATH = $oldPythonPath
    $env:PYTHONDONTWRITEBYTECODE = $oldDontWriteBytecode
}
