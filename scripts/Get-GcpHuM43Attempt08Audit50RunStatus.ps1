param(
    [Parameter(Mandatory = $true)][string]$RunName,
    [string]$ProjectId = 'ofc-solver-485418',
    [string]$Bucket = 'pokerhu-ofc-solver-485418-training',
    [string]$RunDir,
    [string]$DevelopmentRunDir
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $RunDir) { $RunDir = Join-Path $repoRoot "outputs/gcp_runs/$RunName" }
$RunDir = (Resolve-Path $RunDir).Path
$manifest = Get-Content (Join-Path $RunDir 'manifest.json') -Raw | ConvertFrom-Json
if (-not $DevelopmentRunDir) { $DevelopmentRunDir = Join-Path $repoRoot "outputs/gcp_runs/$($manifest.development_run_name)" }
$DevelopmentRunDir = (Resolve-Path $DevelopmentRunDir).Path
$script:Audit50DevelopmentRunDir = $DevelopmentRunDir
$frozenScript = Join-Path $RunDir 'overlay_src/scripts/Get-GcpHuM43Attempt08Audit50RunStatus.ps1'
$frozenCommon = Join-Path $RunDir 'overlay_src/scripts/HuM43Attempt08Audit50Spot.Common.ps1'
if ((Get-FileHash $PSCommandPath -Algorithm SHA256).Hash -ne (Get-FileHash $frozenScript -Algorithm SHA256).Hash) { throw 'Attempt08 audit50 Status script differs from frozen overlay' }
. $frozenCommon
$module = Join-Path $RunDir 'overlay_src/src/ofc_regular/hu_m43_attempt08_audit50_spot.py'
$authorization = Join-Path $RunDir 'launch_authorization.json'
[void](Invoke-M43A8Audit50Python -ModuleFile $module -Arguments @('validate-launch','--run-dir',$RunDir,'--development-run-dir',$DevelopmentRunDir,'--authorization',$authorization) -Label 'validate audit50 status chain' -TimeoutSeconds 3600)

$mirror = Join-Path $RunDir 'status_done_only'
New-Item -ItemType Directory -Path $mirror -Force | Out-Null
$prefix = "gs://$Bucket/runs/$RunName"
$done = [Collections.Generic.List[int]]::new()
$missing = [Collections.Generic.List[int]]::new()
for ($shard = 0; $shard -lt 50; $shard++) {
    $uri = "$prefix/results/shard_$('{0:d3}' -f $shard)/DONE.json"
    $destination = Join-Path $mirror ('DONE-{0:d3}.json' -f $shard)
    if (Test-M43A4GcsObject -Uri $uri -ProjectId $ProjectId) {
        Copy-M43A4RemoteFileExact -Uri $uri -Destination $destination -ProjectId $ProjectId
        [void](Invoke-M43A8Audit50Python -ModuleFile $module -Arguments @('validate-done','--run-dir',$RunDir,'--authorization',$authorization,'--shard',[string]$shard,'--done',$destination) -Label "validate audit50 DONE $shard")
        $done.Add($shard)
    } else {
        Remove-Item $destination -Force -ErrorAction SilentlyContinue
        $missing.Add($shard)
    }
}
if ($done.Count -eq 50) {
    [void](Invoke-M43A8Audit50Python -ModuleFile $module -Arguments @('validate-done-set','--run-dir',$RunDir,'--authorization',$authorization,'--done-root',$mirror) -Label 'validate exact audit50 DONE set')
}
[pscustomobject]@{
    schema = 'hu_m43_attempt08_audit50_done_only_status_v1'
    status = if ($done.Count -eq 50) { 'all_50_done_valid' } else { 'incomplete' }
    run_name = $RunName
    done_count = $done.Count
    missing_count = $missing.Count
    done_shards = @($done)
    missing_shards = @($missing)
    audit_content_opened = $false
    selector_executed = $false
    current_profile_mutated = $false
    runtime_policy_activated = $false
} | ConvertTo-Json -Depth 6
