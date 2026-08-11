param(
    [Parameter(Mandatory = $true)][string]$RunName,
    [string[]]$Shards = @('0-49'),
    [string]$ProjectId = 'ofc-solver-485418',
    [string]$Bucket = 'pokerhu-ofc-solver-485418-training',
    [string]$RunDir
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'HuM43Attempt04Spot.Common.ps1')

function Expand-M43A6StatusSelection {
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
        if ($index -lt 0 -or $index -ge 50) { throw "Shard outside 0..49: $index" }
    }
    return @($selected)
}

function Read-M43A6RemoteJsonExact {
    param([string]$Uri)
    $temporary = Join-Path ([IO.Path]::GetTempPath()) ('m43a6-status-' + [guid]::NewGuid().ToString('N') + '.json')
    try {
        Copy-M43A4RemoteFileExact -Uri $Uri -Destination $temporary -ProjectId $ProjectId
        return (Get-Content -LiteralPath $temporary -Raw | ConvertFrom-Json)
    }
    finally { Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue }
}

function Assert-M43A6Done {
    param($Done, $Spec, $Manifest, [string]$ManifestSha256)
    if ($Done.schema -ne 'hu_m43_attempt06_spot_done_v1' -or
        $Done.status -ne 'complete' -or
        [string]$Done.run_name -ne $RunName -or
        [int]$Done.shard -ne [int]$Spec.shard -or
        [int]$Done.root_index -ne [int]$Spec.root_index -or
        [int64]$Done.hand_seed -ne [int64]$Spec.hand_seed -or
        [string]$Done.root_profile -ne [string]$Spec.root_profile -or
        [int]$Done.roots -ne 1 -or
        [string]$Done.output_prefix -ne [string]$Spec.output_prefix -or
        [string]$Done.manifest_sha256 -ne $ManifestSha256 -or
        [string]$Done.schedule_sha256 -ne [string]$Manifest.schedule_sha256 -or
        [string]$Done.source_sha256 -ne [string]$Manifest.source_zip_sha256 -or
        [string]$Done.startup_sha256 -ne [string]$Manifest.startup_sha256 -or
        [string]$Done.plan_sha256 -ne [string]$Manifest.plan_sha256 -or
        [string]$Done.status_sha256 -ne [string]$Manifest.status_sha256 -or
        [string]$Done.model_sha256 -ne [string]$Manifest.model_sha256 -or
        [string]$Done.source_closure_sha256 -ne [string]$Manifest.source_closure_sha256 -or
        [string]$Done.source_model_manifest_sha256 -ne [string]$Manifest.source_model_manifest_sha256 -or
        [string]$Done.source_native_manifest_sha256 -ne [string]$Manifest.source_native_manifest_sha256 -or
        $Done.teacher_values_are_realized_match_ev -ne $false -or
        $Done.current_profile_mutated -ne $false -or
        $Done.runtime_policy_activated -ne $false) {
        throw "Attempt06 DONE identity changed for shard $($Spec.shard)"
    }
    foreach ($name in @(
        'input_sha256', 'output_sha256', 'checkpoint_sha256', 'heartbeat_sha256',
        'generator_summary_sha256', 'run_log_sha256',
        'global_consumption_marker_sha256', 'root_consumption_claim_sha256'
    )) {
        Assert-M43A4Sha256 ([string]$Done.$name) "Attempt06 DONE $name"
    }
}

if ($RunName -notmatch '^[a-z0-9][a-z0-9-]{2,126}[a-z0-9]$') { throw 'unsafe RunName' }
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $RunDir) { $RunDir = Join-Path (Join-Path $repoRoot 'outputs/gcp_runs') $RunName }
$RunDir = Resolve-M43A4Path -Path $RunDir -Root $repoRoot -Label 'Attempt06 run directory'
Assert-M43A4UnderRoot -Path $RunDir -Root $repoRoot -Label 'Attempt06 run directory'
$manifestPath = Join-Path $RunDir 'manifest.json'
$schedulePath = Join-Path $RunDir 'shards_manifest.jsonl'
foreach ($path in @($manifestPath, $schedulePath)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Attempt06 closure missing: $path" }
}
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$manifestSha256 = Get-M43A4Sha256 $manifestPath
if ($manifest.schema -ne 'hu_m43_attempt06_spot_package_manifest_v1' -or
    $manifest.status -ne 'frozen_package_only_no_fresh_content' -or
    [string]$manifest.run_name -ne $RunName -or
    [int]$manifest.total_shards -ne 50 -or [int]$manifest.roots_per_shard -ne 1 -or
    [string]$manifest.schedule_sha256 -ne (Get-M43A4Sha256 $schedulePath)) {
    throw 'Attempt06 local package closure changed'
}
$specs = @([IO.File]::ReadLines($schedulePath) | Where-Object { $_.Trim() } | ForEach-Object { $_ | ConvertFrom-Json })
if ($specs.Count -ne 50 -or @($specs | Where-Object { [int]$_.roots -ne 1 }).Count -ne 0) {
    throw 'Attempt06 schedule is not exactly fifty one-root shards'
}
$selected = @(Expand-M43A6StatusSelection -Values $Shards)
if ($selected.Count -eq 0) { throw 'no status shards selected' }
$selectedSet = [Collections.Generic.HashSet[int]]::new()
foreach ($index in $selected) { [void]$selectedSet.Add($index) }
$selectedSpecs = @($specs | Where-Object { $selectedSet.Contains([int]$_.shard) })
if ($selectedSpecs.Count -ne $selected.Count) { throw 'Attempt06 selected schedule mapping changed' }

$prefix = "gs://$Bucket/runs/$RunName"
$temporaryRoot = Join-Path ([IO.Path]::GetTempPath()) ('m43a6-status-closure-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $temporaryRoot | Out-Null
try {
    Copy-M43A4RemoteFileExact -Uri "$prefix/manifest.json" `
        -Destination (Join-Path $temporaryRoot 'manifest.json') -ProjectId $ProjectId `
        -ExpectedSha256 $manifestSha256
    Copy-M43A4RemoteFileExact -Uri "$prefix/source/shards_manifest.jsonl" `
        -Destination (Join-Path $temporaryRoot 'shards_manifest.jsonl') -ProjectId $ProjectId `
        -ExpectedSha256 ([string]$manifest.schedule_sha256)
}
finally { Remove-Item -LiteralPath $temporaryRoot -Recurse -Force -ErrorAction SilentlyContinue }

# Only exact DONE metadata for the explicitly selected shards is addressed.
$rows = foreach ($spec in $selectedSpecs) {
    $doneUri = "$prefix/results/$($spec.output_prefix)/DONE.json"
    $present = Test-M43A4GcsObject -Uri $doneUri -ProjectId $ProjectId
    if ($present) {
        $done = Read-M43A6RemoteJsonExact -Uri $doneUri
        Assert-M43A6Done -Done $done -Spec $spec -Manifest $manifest -ManifestSha256 $manifestSha256
    }
    [pscustomobject][ordered]@{
        shard = [int]$spec.shard
        root_index = [int]$spec.root_index
        root_profile = [string]$spec.root_profile
        state = $(if ($present) { 'complete_verified_done' } else { 'not_complete' })
    }
}
$complete = @($rows | Where-Object { $_.state -eq 'complete_verified_done' }).Count
[pscustomobject][ordered]@{
    schema = 'hu_m43_attempt06_spot_status_result_v1'
    run_name = $RunName
    selected_shards = $selected.Count
    complete_shards = $complete
    incomplete_shards = $selected.Count - $complete
    all_done_markers_verified = ($complete -eq $selected.Count)
    manifest_sha256 = $manifestSha256
    schedule_sha256 = [string]$manifest.schedule_sha256
    teacher_payload_downloaded = $false
    root_payload_downloaded = $false
    exact_done_objects_only = $true
    shards = $rows
    current_profile_mutated = $false
    runtime_policy_activated = $false
} | ConvertTo-Json -Depth 8
