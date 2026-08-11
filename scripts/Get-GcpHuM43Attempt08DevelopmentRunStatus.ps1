param(
    [Parameter(Mandatory = $true)][string]$RunName,
    [string]$ProjectId = 'ofc-solver-485418',
    [string]$Bucket = 'pokerhu-ofc-solver-485418-training',
    [string]$RunDir
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if ($RunName -notmatch '^[a-z0-9][a-z0-9-]{2,126}[a-z0-9]$') { throw 'Unsafe Attempt08 RunName' }
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $RunDir) { $RunDir = Join-Path $repoRoot "outputs/gcp_runs/$RunName" }
$RunDir = [IO.Path]::GetFullPath($RunDir)
$expected = [IO.Path]::GetFullPath((Join-Path $repoRoot "outputs/gcp_runs/$RunName"))
if (-not [string]::Equals($RunDir, $expected, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Attempt08 status RunDir must be the exact repository run directory'
}
$frozenScript = Join-Path $RunDir 'package_src/scripts/Get-GcpHuM43Attempt08DevelopmentRunStatus.ps1'
$frozenCommon = Join-Path $RunDir 'package_src/scripts/HuM43Attempt08Spot.Common.ps1'
foreach ($path in @($frozenScript,$frozenCommon,(Join-Path $RunDir 'package_src/scripts/HuM43Attempt04Spot.Common.ps1'))) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Attempt08 frozen status source is missing: $path" }
}
if ((Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash -ne
    (Get-FileHash -LiteralPath $frozenScript -Algorithm SHA256).Hash) {
    throw 'Attempt08 status script differs from its frozen packaged copy'
}
. $frozenCommon
$manifestPath = Join-Path $RunDir 'manifest.json'
$authorizationPath = Join-Path $RunDir 'launch_authorization.json'
$schedulePath = Join-Path $RunDir 'shards_manifest.jsonl'
foreach ($path in @($manifestPath,$authorizationPath,$schedulePath)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Attempt08 package file is missing: $path" }
}
$frozenModuleRoot = Join-Path $RunDir 'package_src'
$validation = Invoke-M43A8Python -RepoRoot $frozenModuleRoot -Arguments @(
    '-B','-m','ofc_regular.hu_m43_attempt08_spot','validate-launch',
    '--run-dir',$RunDir,'--authorization',$authorizationPath
) -Label 'validate Attempt08 status package' -TimeoutSeconds 3600
if (-not $validation.Trim()) { throw 'Attempt08 package validation returned no result' }

$schedule = @(Get-Content -LiteralPath $schedulePath | Where-Object { $_ } | ForEach-Object { $_ | ConvertFrom-Json })
if ($schedule.Count -ne 200) { throw 'Attempt08 status requires exact 200-shard schedule' }
$mirror = Join-Path $RunDir ('.status-done-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $mirror | Out-Null
$done = [Collections.Generic.List[int]]::new()
$missing = [Collections.Generic.List[int]]::new()
try {
    foreach ($spec in $schedule) {
        $shard = [int]$spec.shard
        if ($shard -lt 0 -or $shard -ge 200 -or [int]$spec.root_index -ne $shard) {
            throw 'Attempt08 status schedule mapping changed'
        }
        $uri = "gs://$Bucket/runs/$RunName/results/$($spec.output_prefix)/DONE.json"
        if (-not (Test-M43A4GcsObject -Uri $uri -ProjectId $ProjectId)) {
            $missing.Add($shard); continue
        }
        $destination = Join-Path $mirror ('DONE-{0:d3}.json' -f $shard)
        Copy-M43A4RemoteFileExact -Uri $uri -Destination $destination -ProjectId $ProjectId
        [void](Invoke-M43A8Python -RepoRoot $frozenModuleRoot -Arguments @(
            '-B','-m','ofc_regular.hu_m43_attempt08_spot','validate-done',
            '--input',$destination,'--shard',"$shard",'--run-dir',$RunDir,
            '--authorization',$authorizationPath
        ) -Label "validate Attempt08 DONE $shard" -TimeoutSeconds 3600)
        $done.Add($shard)
    }
    $allDone = $done.Count -eq 200
    if ($allDone) {
        [void](Invoke-M43A8Python -RepoRoot $frozenModuleRoot -Arguments @(
            '-B','-m','ofc_regular.hu_m43_attempt08_spot','validate-done-set',
            '--done-root',$mirror,'--run-dir',$RunDir,'--authorization',$authorizationPath
        ) -Label 'validate exact Attempt08 DONE set' -TimeoutSeconds 3600)
    }
    $restartWaves = [Collections.Generic.List[object]]::new()
    for ($offset = 0; $offset -lt $missing.Count; $offset += 25) {
        $last = [Math]::Min($offset + 24, $missing.Count - 1)
        $restartWaves.Add(@($missing[$offset..$last]))
    }
    [pscustomobject]@{
        schema = 'hu_m43_attempt08_development_done_only_status_v1'
        status = if ($allDone) { 'all_200_done_verified' } else { 'incomplete' }
        run_name = $RunName
        done_count = $done.Count
        missing_count = $missing.Count
        done_shards = @($done)
        missing_shards = @($missing)
        restart_waves_max_25 = @($restartWaves)
        all_done_markers_verified = $allDone
        result_content_opened = $false
        selector_executed = $false
        current_profile_mutated = $false
        runtime_policy_activated = $false
    } | ConvertTo-Json -Depth 8
}
finally {
    $resolvedMirror = [IO.Path]::GetFullPath($mirror)
    if (-not $resolvedMirror.StartsWith($expected + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Attempt08 status mirror escaped the run directory'
    }
    Remove-Item -LiteralPath $resolvedMirror -Recurse -Force -ErrorAction SilentlyContinue
}
