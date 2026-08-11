param(
    [Parameter(Mandatory = $true)][string]$RunName,
    [string]$ProjectId = 'ofc-solver-485418',
    [string]$Bucket = 'pokerhu-ofc-solver-485418-training',
    [string]$RunDir,
    [string]$SpotAuthorizationPath,
    [string[]]$Shards = @('0-99')
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'HuM43Attempt04Spot.Common.ps1')

function Expand-M43A7StatusSelection {
    param([string[]]$Values)
    $selected = [Collections.Generic.SortedSet[int]]::new()
    foreach ($raw in $Values) {
        foreach ($token in ([string]$raw -split ',')) {
            $value = $token.Trim()
            if (-not $value) { continue }
            if ($value -match '^(\d+)-(\d+)$') {
                $first = [int]$Matches[1]; $last = [int]$Matches[2]
                if ($first -gt $last) { throw "Invalid shard range: $value" }
                for ($index = $first; $index -le $last; $index++) { [void]$selected.Add($index) }
            }
            elseif ($value -match '^\d+$') { [void]$selected.Add([int]$value) }
            else { throw "Invalid shard selection: $value" }
        }
    }
    foreach ($index in $selected) {
        if ($index -lt 0 -or $index -ge 100) { throw "Shard outside 0..99: $index" }
    }
    return @($selected)
}

if ($RunName -notmatch '^[a-z0-9][a-z0-9-]{2,126}[a-z0-9]$') { throw 'unsafe RunName' }
if (-not $SpotAuthorizationPath) { throw 'SpotAuthorizationPath is required' }
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $RunDir) { $RunDir = Join-Path (Join-Path $repoRoot 'outputs/gcp_runs') $RunName }
$RunDir = Resolve-M43A4Path -Path $RunDir -Root $repoRoot -Label 'Attempt07 run directory'
Assert-M43A4UnderRoot -Path $RunDir -Root $repoRoot -Label 'Attempt07 run directory'
$manifestPath = Resolve-M43A4Path -Path (Join-Path $RunDir 'manifest.json') -Root $repoRoot -Label 'Attempt07 manifest' -RequireFile
$schedulePath = Resolve-M43A4Path -Path (Join-Path $RunDir 'shards_manifest.jsonl') -Root $repoRoot -Label 'Attempt07 schedule' -RequireFile
$authorizationPath = Resolve-M43A4Path -Path $SpotAuthorizationPath -Root $repoRoot -Label 'Attempt07 Spot authorization' -RequireFile
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$manifestSha256 = Get-M43A4Sha256 $manifestPath
$authorizationSha256 = Get-M43A4Sha256 $authorizationPath
if ($manifest.schema -ne 'hu_m43_attempt07_development_spot_package_v1' -or
    $manifest.status -ne 'frozen_package_only_no_root_opened' -or
    [string]$manifest.run_name -ne $RunName -or [int]$manifest.total_shards -ne 100 -or
    [string]$manifest.schedule_sha256 -ne (Get-M43A4Sha256 $schedulePath) -or
    $manifest.current_profile_mutated -ne $false -or $manifest.runtime_policy_activated -ne $false) {
    throw 'Attempt07 local status closure changed'
}
$selected = @(Expand-M43A7StatusSelection -Values $Shards)
$prefix = "gs://$Bucket/runs/$RunName"
$rows = [Collections.Generic.List[object]]::new()
$temporaryRoot = Join-Path ([IO.Path]::GetTempPath()) ('m43a7-status-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $temporaryRoot | Out-Null
try {
    foreach ($shard in $selected) {
        $outputPrefix = 'shard_{0:D3}' -f $shard
        $doneUri = "$prefix/results/$outputPrefix/DONE.json"
        if (-not (Test-M43A4GcsObject -Uri $doneUri -ProjectId $ProjectId)) {
            $rows.Add([pscustomobject][ordered]@{
                shard = $shard; status = 'not_done'; elapsed_seconds = $null; peak_rss_bytes = $null
            })
            continue
        }
        $donePath = Join-Path $temporaryRoot ("DONE-{0:D3}.json" -f $shard)
        Copy-M43A4RemoteFileExact -Uri $doneUri -Destination $donePath -ProjectId $ProjectId
        $done = Get-Content -LiteralPath $donePath -Raw | ConvertFrom-Json
        if ($done.schema -ne 'hu_m43_attempt07_development_spot_done_v1' -or
            $done.status -ne 'complete' -or [string]$done.run_name -ne $RunName -or
            [int]$done.shard -ne $shard -or [int]$done.root_index -ne $shard -or
            [string]$done.output_prefix -ne $outputPrefix -or
            [string]$done.manifest_sha256 -ne $manifestSha256 -or
            [string]$done.authorization_sha256 -ne $authorizationSha256 -or
            $done.current_profile_mutated -ne $false -or
            $done.runtime_policy_activated -ne $false) {
            throw "Attempt07 DONE identity changed for shard $shard"
        }
        foreach ($field in @(
            'authorization_sha256','global_claim_sha256','root_claim_sha256',
            'output_sha256','checkpoint_sha256','heartbeat_sha256',
            'generator_summary_sha256','run_log_sha256','config_sha256'
        )) {
            Assert-M43A4Sha256 ([string]$done.$field) "Attempt07 DONE $field"
        }
        $elapsed = [double]$done.elapsed_seconds
        $rss = [int64]$done.peak_rss_bytes
        if ([double]::IsNaN($elapsed) -or [double]::IsInfinity($elapsed) -or
            $elapsed -lt 0.0 -or $rss -lt 0) {
            throw "Attempt07 DONE operational metadata changed for shard $shard"
        }
        $rows.Add([pscustomobject][ordered]@{
            shard = $shard; status = 'done'; elapsed_seconds = $elapsed; peak_rss_bytes = $rss
        })
    }
}
finally { Remove-Item -LiteralPath $temporaryRoot -Recurse -Force -ErrorAction SilentlyContinue }

$doneRows = @($rows | Where-Object { $_.status -eq 'done' })
$isFullSelection = ($selected.Count -eq 100 -and $selected[0] -eq 0 -and $selected[99] -eq 99)
$all100Done = ($isFullSelection -and $doneRows.Count -eq 100)
[pscustomobject][ordered]@{
    schema = 'hu_m43_attempt07_development_spot_status_result_v1'
    status = $(if ($all100Done) { 'all_100_done' } elseif ($doneRows.Count -eq $selected.Count) { 'selected_shards_done' } else { 'incomplete' })
    run_name = $RunName
    selected_count = $selected.Count
    done_count = $doneRows.Count
    all_100_done = $all100Done
    elapsed_seconds_sum = [double](($doneRows | Measure-Object elapsed_seconds -Sum).Sum)
    elapsed_seconds_max = [double](($doneRows | Measure-Object elapsed_seconds -Maximum).Maximum)
    peak_rss_bytes_max = [int64](($doneRows | Measure-Object peak_rss_bytes -Maximum).Maximum)
    rows = @($rows)
    result_content_addressed = $false
    selector_executed = $false
    current_profile_mutated = $false
    runtime_policy_activated = $false
} | ConvertTo-Json -Depth 8
