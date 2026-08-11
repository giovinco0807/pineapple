param(
    [string]$ProjectId = "ofc-solver-485418",
    [string]$Bucket = "pokerhu-ofc-solver-485418-training",
    [Parameter(Mandatory = $true)]
    [string]$RunName,
    [int]$MaxMissingToShow = 50
)

$ErrorActionPreference = "Stop"
$statusScript = Join-Path $PSScriptRoot "Get-GcpHuTurn1PilotRunStatus.ps1"
& $statusScript `
    -ProjectId $ProjectId `
    -Bucket $Bucket `
    -RunName $RunName `
    -MaxMissingToShow $MaxMissingToShow
if ($LASTEXITCODE -ne 0) {
    throw "HU T0 GCP pilot status failed with exit code $LASTEXITCODE"
}
