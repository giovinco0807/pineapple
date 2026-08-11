param(
    [string]$ProjectId = "ofc-solver-485418",
    [string]$Bucket = "pokerhu-ofc-solver-485418-training",
    [Parameter(Mandatory = $true)]
    [string]$RunName,
    [string]$DownloadDir,
    [string]$OutputDir = "outputs/hu_turn0_stage1_pilot/gcp_aggregate",
    [switch]$AllowPartial
)

$ErrorActionPreference = "Stop"
$receiver = Join-Path $PSScriptRoot "Receive-GcpHuTurn1PilotRun.ps1"
$arguments = @{
    ProjectId = $ProjectId
    Bucket = $Bucket
    RunName = $RunName
    OutputDir = $OutputDir
    AggregatorModule = "ofc_regular.aggregate_hu_turn0_pilot"
    MergedOutputName = "hu_turn0_stage1_pilot.jsonl"
    ShardSummariesName = "hu_turn0_stage1_pilot_summaries.jsonl"
    AllowPartial = [bool]$AllowPartial
}
if ($DownloadDir) {
    $arguments["DownloadDir"] = $DownloadDir
}

& $receiver @arguments
if ($LASTEXITCODE -ne 0) {
    throw "HU T0 GCP pilot receive failed with exit code $LASTEXITCODE"
}
