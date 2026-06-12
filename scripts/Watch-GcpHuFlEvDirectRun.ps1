param(
    [Parameter(Mandatory = $true)]
    [string]$RunName,
    [int]$IntervalSeconds = 300,
    [string]$OutputDir = "outputs/fl_ev_direct_hu_gcp",
    [string]$LogPath,
    [switch]$ReceiveOnComplete
)

$ErrorActionPreference = "Continue"

if (-not $LogPath) {
    $LogPath = "outputs/gcp_runs/$RunName/watch_status.jsonl"
}
New-Item -ItemType Directory -Force -Path (Split-Path $LogPath -Parent) | Out-Null

while ($true) {
    $checkedAt = (Get-Date).ToUniversalTime().ToString("o")
    try {
        $raw = & "$PSScriptRoot\Get-GcpHuFlEvDirectRunStatus.ps1" -RunName $RunName
        $status = $raw | ConvertFrom-Json
        $record = [ordered]@{
            checked_at = $checkedAt
            run_name = $RunName
            expected_shards = $status.expected_shards
            completed_shards = $status.completed_shards
            missing_count = $status.missing_count
            failed_shards = $status.failed_shards
            running_instances = $status.running_instances
        }
        Add-Content -Path $LogPath -Value (($record | ConvertTo-Json -Compress -Depth 8))

        if ([int]$status.completed_shards -eq [int]$status.expected_shards -and [int]$status.expected_shards -gt 0) {
            if ($ReceiveOnComplete) {
                $receiveRaw = & "$PSScriptRoot\Receive-GcpHuFlEvDirectRun.ps1" -RunName $RunName -OutputDir $OutputDir
                Add-Content -Path $LogPath -Value (([ordered]@{
                    checked_at = (Get-Date).ToUniversalTime().ToString("o")
                    run_name = $RunName
                    receive = "complete"
                    output_dir = $OutputDir
                    raw = $receiveRaw
                } | ConvertTo-Json -Compress -Depth 8))
            }
            break
        }

        if ($status.failed_shards -and @($status.failed_shards).Count -gt 0) {
            Add-Content -Path $LogPath -Value (([ordered]@{
                checked_at = (Get-Date).ToUniversalTime().ToString("o")
                run_name = $RunName
                stop_reason = "failed_shards"
                failed_shards = $status.failed_shards
            } | ConvertTo-Json -Compress -Depth 8))
            break
        }
    }
    catch {
        Add-Content -Path $LogPath -Value (([ordered]@{
            checked_at = $checkedAt
            run_name = $RunName
            error = $_.Exception.Message
        } | ConvertTo-Json -Compress -Depth 8))
    }

    Start-Sleep -Seconds $IntervalSeconds
}
