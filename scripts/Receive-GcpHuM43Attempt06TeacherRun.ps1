param(
    [Parameter(Mandatory = $true)][string]$RunName,
    [Parameter(Mandatory = $true)][string]$AuditOpenAuthorizationPath,
    [Parameter(Mandatory = $true)][string]$AuditConsumptionMarkerPath,
    [string]$ProjectId = 'ofc-solver-485418',
    [string]$Bucket = 'pokerhu-ofc-solver-485418-training',
    [string]$RunDir,
    [string]$OutputDir,
    [ValidateRange(1, 8)][int]$MaxParallel = 8,
    [switch]$ResumeClaim
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'HuM43Attempt04Spot.Common.ps1')

function Invoke-M43A6ReceivePython {
    param([string[]]$Arguments, [string]$Label, [int]$TimeoutSeconds = 300)
    $pythonPath = (Get-Command python -ErrorAction Stop | Select-Object -First 1).Source
    $result = Invoke-M43A4ProcessBounded `
        -FilePath $pythonPath -Arguments $Arguments -Label $Label `
        -TimeoutSeconds $TimeoutSeconds `
        -Environment @{ PYTHONPATH = (Join-Path $script:RepoRoot 'src') }
    if ($result.timed_out -or $result.exit_code -ne 0) {
        throw "$Label failed: $($result.stderr)$($result.stdout)"
    }
    return [string]$result.stdout
}

function Assert-M43A6ReceiveAuthorization {
    param($Authorization, [string]$ManifestSha256, [string]$GlobalMarkerSha256)
    if ($Authorization.schema -ne 'hu_m43_attempt06_audit_open_authorization_v1' -or
        $Authorization.status -ne 'authorized_to_open_complete_search_quality_audit_once' -or
        [string]$Authorization.run_name -ne $RunName -or
        [string]$Authorization.manifest_sha256 -ne $ManifestSha256 -or
        [string]$Authorization.global_consumption_marker_sha256 -ne $GlobalMarkerSha256 -or
        [int]$Authorization.expected_shards -ne 50 -or
        [int]$Authorization.expected_roots -ne 50 -or
        $Authorization.all_done_markers_verified -ne $true -or
        $Authorization.fit_allowed -ne $false -or
        $Authorization.threshold_selection_allowed -ne $false -or
        $Authorization.current_profile_mutated -ne $false -or
        $Authorization.runtime_policy_activated -ne $false) {
        throw 'Attempt06 audit-open authorization identity changed'
    }
}

function Read-M43A6ExistingClaim {
    param(
        [string]$Path, [string]$AuthorizationSha256,
        [string]$ManifestSha256, [string]$GlobalMarkerSha256
    )
    $claim = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    if ($claim.schema -ne 'hu_m43_attempt06_audit_output_consumption_v1' -or
        $claim.status -ne 'consumed_before_any_result_teacher_or_root_read' -or
        [string]$claim.run_name -ne $RunName -or
        [string]$claim.authorization_file_sha256 -ne $AuthorizationSha256 -or
        [string]$claim.manifest_sha256 -ne $ManifestSha256 -or
        [string]$claim.global_consumption_marker_sha256 -ne $GlobalMarkerSha256 -or
        $claim.result_objects_addressed_when_claimed -ne $false -or
        $claim.fit_performed -ne $false -or
        $claim.threshold_selected -ne $false -or
        $claim.current_profile_mutated -ne $false -or
        $claim.runtime_policy_activated -ne $false) {
        throw 'existing Attempt06 audit-output claim does not match this run'
    }
    return $claim
}

function Claim-M43A6RemoteAuditOutputOnce {
    param(
        [Parameter(Mandatory = $true)][string]$LocalClaimPath,
        [Parameter(Mandatory = $true)][string]$RemoteClaimUri,
        [Parameter(Mandatory = $true)][bool]$AllowExisting
    )
    $upload = Invoke-M43A4GcloudProcess `
        -Arguments @(
            'storage', 'cp', $LocalClaimPath, $RemoteClaimUri,
            '--project', $ProjectId, '--if-generation-match=0'
        ) -TimeoutSeconds 300 -Label 'claim remote Attempt06 audit output once'
    if (-not $upload.timed_out -and $upload.exit_code -eq 0) { return 'owner' }
    $existing = Join-Path ([IO.Path]::GetTempPath()) (
        'm43a6-audit-output-claim-' + [guid]::NewGuid().ToString('N') + '.json'
    )
    try {
        Copy-M43A4RemoteFileExact `
            -Uri $RemoteClaimUri -Destination $existing -ProjectId $ProjectId
        if ((Get-M43A4Sha256 $existing) -ne (Get-M43A4Sha256 $LocalClaimPath)) {
            throw 'remote Attempt06 audit-output claim identity differs'
        }
        if (-not $AllowExisting) {
            throw 'remote Attempt06 audit output is already consumed on another host'
        }
        return 'existing'
    }
    finally { Remove-Item -LiteralPath $existing -Force -ErrorAction SilentlyContinue }
}

function Publish-M43A6ReceiveDirectoryAtomic {
    param([string]$Source, [string]$Destination)
    $sourceFull = [IO.Path]::GetFullPath($Source)
    $destinationFull = [IO.Path]::GetFullPath($Destination)
    if (Test-Path -LiteralPath $destinationFull) {
        throw "Attempt06 receive destination already exists: $destinationFull"
    }
    [IO.Directory]::Move($sourceFull, $destinationFull)
}

if ($RunName -notmatch '^[a-z0-9][a-z0-9-]{2,126}[a-z0-9]$') { throw 'unsafe RunName' }
$script:RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $RunDir) { $RunDir = Join-Path (Join-Path $script:RepoRoot 'outputs/gcp_runs') $RunName }
if (-not $OutputDir) {
    $OutputDir = Join-Path $script:RepoRoot "outputs/hu_joint_policy/m43_attempt06_search_quality/$RunName"
}
$RunDir = Resolve-M43A4Path -Path $RunDir -Root $script:RepoRoot -Label 'Attempt06 run directory'
$OutputDir = Resolve-M43A4Path -Path $OutputDir -Root $script:RepoRoot -Label 'Attempt06 receive output'
$authorizationPath = Resolve-M43A4Path -Path $AuditOpenAuthorizationPath -Root $script:RepoRoot -Label 'Attempt06 audit-open authorization' -RequireFile
$claimPath = Resolve-M43A4Path -Path $AuditConsumptionMarkerPath -Root $script:RepoRoot -Label 'Attempt06 audit-output marker'
foreach ($binding in @(
    @($RunDir, 'Attempt06 run directory'),
    @($OutputDir, 'Attempt06 receive output'),
    @($claimPath, 'Attempt06 audit-output marker')
)) {
    Assert-M43A4UnderRoot -Path ([string]$binding[0]) -Root $script:RepoRoot -Label ([string]$binding[1])
}
$canonicalClaimPath = [IO.Path]::GetFullPath((Join-Path $script:RepoRoot `
    'outputs/hu_joint_policy/m43_attempt06_search_quality/M43_ATTEMPT06_AUDIT_CONSUMED.json'))
$claimFull = [IO.Path]::GetFullPath($claimPath)
if (-not [string]::Equals(
    $claimFull, $canonicalClaimPath, [StringComparison]::OrdinalIgnoreCase
)) {
    throw 'AuditConsumptionMarkerPath must equal the single canonical Attempt06 marker path'
}
$outputFull = [IO.Path]::GetFullPath($OutputDir).TrimEnd(
    [IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar
)
if ($claimFull.StartsWith(
    $outputFull + [IO.Path]::DirectorySeparatorChar,
    [StringComparison]::OrdinalIgnoreCase
)) {
    throw 'Attempt06 audit consumption marker must remain outside immutable receive output'
}
if (Test-Path -LiteralPath $OutputDir) {
    throw 'Attempt06 receive output already exists; immutable output is never overwritten'
}
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
    [int]$manifest.native_batch_threads -ne 4 -or
    [string]$manifest.schedule_sha256 -ne (Get-M43A4Sha256 $schedulePath)) {
    throw 'Attempt06 local receive closure changed'
}
$specs = @([IO.File]::ReadLines($schedulePath) | Where-Object { $_.Trim() } | ForEach-Object { $_ | ConvertFrom-Json })
if ($specs.Count -ne 50) { throw 'Attempt06 receive requires all fifty shards' }
$prefix = "gs://$Bucket/runs/$RunName"
$stagingRoot = $null

# This remote marker has no cards or teacher values and was created before any
# worker opened a root.  It is the only run-state object read before the local
# one-shot result claim.
$temporaryGlobalMarker = Join-Path ([IO.Path]::GetTempPath()) ('m43a6-global-' + [guid]::NewGuid().ToString('N') + '.json')
try {
    Copy-M43A4RemoteFileExact `
        -Uri "$prefix/fresh_boundary/CONSUMED.json" `
        -Destination $temporaryGlobalMarker -ProjectId $ProjectId
    $globalMarker = Get-Content -LiteralPath $temporaryGlobalMarker -Raw | ConvertFrom-Json
    if ($globalMarker.schema -ne 'hu_m43_attempt06_global_consumption_marker_v1' -or
        $globalMarker.status -ne 'consumed_before_any_fresh_root_content_read' -or
        [string]$globalMarker.run_name -ne $RunName -or
        [string]$globalMarker.manifest_sha256 -ne $manifestSha256 -or
        [string]$globalMarker.schedule_sha256 -ne [string]$manifest.schedule_sha256 -or
        [string]$globalMarker.plan_sha256 -ne [string]$manifest.plan_sha256 -or
        [string]$globalMarker.source_sha256 -ne [string]$manifest.source_zip_sha256 -or
        [string]$globalMarker.startup_sha256 -ne [string]$manifest.startup_sha256 -or
        [string]$globalMarker.status_sha256 -ne [string]$manifest.status_sha256 -or
        [string]$globalMarker.source_closure_sha256 -ne [string]$manifest.source_closure_sha256 -or
        [string]$globalMarker.model_sha256 -ne [string]$manifest.model_sha256 -or
        [string]$globalMarker.source_model_manifest_sha256 -ne [string]$manifest.source_model_manifest_sha256 -or
        [string]$globalMarker.source_native_manifest_sha256 -ne [string]$manifest.source_native_manifest_sha256 -or
        [int]$globalMarker.native_batch_threads -ne [int]$manifest.native_batch_threads -or
        [int]$globalMarker.roots -ne 50 -or [int]$globalMarker.shards -ne 50 -or
        [int]$globalMarker.roots_per_shard -ne 1) {
        throw 'Attempt06 global fresh-boundary marker changed'
    }
    $globalMarkerSha256 = Get-M43A4Sha256 $temporaryGlobalMarker
    $authorization = Get-Content -LiteralPath $authorizationPath -Raw | ConvertFrom-Json
    Assert-M43A6ReceiveAuthorization `
        -Authorization $authorization -ManifestSha256 $manifestSha256 `
        -GlobalMarkerSha256 $globalMarkerSha256
    $authorizationSha256 = Get-M43A4Sha256 $authorizationPath

    $localClaimExisted = Test-Path -LiteralPath $claimPath
    if ($localClaimExisted) {
        if (-not $ResumeClaim) { throw 'Attempt06 audit output is already claimed; use ResumeClaim for this exact run only' }
        $claim = Read-M43A6ExistingClaim `
            -Path $claimPath -AuthorizationSha256 $authorizationSha256 `
            -ManifestSha256 $manifestSha256 -GlobalMarkerSha256 $globalMarkerSha256
    }
    else {
        if ($ResumeClaim) { throw 'ResumeClaim requires an existing canonical Attempt06 claim' }
        $claimPayload = [ordered]@{
            schema = 'hu_m43_attempt06_audit_output_consumption_v1'
            status = 'consumed_before_any_result_teacher_or_root_read'
            run_name = $RunName
            authorization_file_sha256 = $authorizationSha256
            manifest_sha256 = $manifestSha256
            schedule_sha256 = [string]$manifest.schedule_sha256
            global_consumption_marker_sha256 = $globalMarkerSha256
            expected_shards = 50
            expected_roots = 50
            result_objects_addressed_when_claimed = $false
            fit_performed = $false
            threshold_selected = $false
            current_profile_mutated = $false
            runtime_policy_activated = $false
        }
        Write-M43A4Utf8CreateNew -Path $claimPath -Text (($claimPayload | ConvertTo-Json -Depth 6) + "`n")
        $claim = Read-M43A6ExistingClaim `
            -Path $claimPath -AuthorizationSha256 $authorizationSha256 `
            -ManifestSha256 $manifestSha256 -GlobalMarkerSha256 $globalMarkerSha256
    }

    # A second immutable run-scoped claim closes the cross-host gap.  A newly
    # created local claim may only own a previously absent remote object;
    # byte-identical existing remote claims are accepted solely for an explicit
    # resume backed by the already-existing canonical local marker.
    $remoteAuditOutputClaimUri = "$prefix/audit_output_boundary/CONSUMED.json"
    $remoteAuditOutputClaimState = Claim-M43A6RemoteAuditOutputOnce `
        -LocalClaimPath $claimPath -RemoteClaimUri $remoteAuditOutputClaimUri `
        -AllowExisting ($localClaimExisted -and [bool]$ResumeClaim)

    # Result object paths are formed only after the irreversible local claim.
    $stagingRoot = $OutputDir + '.staging-' + [guid]::NewGuid().ToString('N')
    New-Item -ItemType Directory -Path $stagingRoot | Out-Null
    $closureDir = Join-Path $stagingRoot 'closure'
    $shardsDir = Join-Path $stagingRoot 'shards'
    $mergedDir = Join-Path $stagingRoot 'merged'
    New-Item -ItemType Directory -Path $closureDir, $shardsDir, $mergedDir | Out-Null
    Copy-Item -LiteralPath $manifestPath -Destination (Join-Path $closureDir 'manifest.json')
    Copy-Item -LiteralPath $schedulePath -Destination (Join-Path $closureDir 'shards_manifest.jsonl')
    foreach ($name in @(
        'source_closure_manifest.json',
        'hu_joint_policy_m43_attempt06.json',
        'hu_joint_policy_m43_attempt06_status.json',
        'startup_hu_m43_attempt06_teacher.sh'
    )) {
        $source = Join-Path $RunDir $name
        if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
            throw "Attempt06 receive package closure is missing: $source"
        }
        Copy-Item -LiteralPath $source -Destination (Join-Path $closureDir $name)
    }
    Copy-Item -LiteralPath $authorizationPath -Destination (Join-Path $closureDir 'audit_open_authorization.json')
    Copy-Item -LiteralPath $claimPath -Destination (Join-Path $closureDir 'audit_consumption_marker.json')
    Copy-Item -LiteralPath $temporaryGlobalMarker -Destination (Join-Path $closureDir 'global_consumption_marker.json')

    $copyJobs = @()
    foreach ($spec in $specs) {
        $shardDir = Join-Path $shardsDir ([string]$spec.output_prefix)
        New-Item -ItemType Directory -Path $shardDir | Out-Null
        $resultPrefix = "$prefix/results/$($spec.output_prefix)"
        Copy-M43A4RemoteFileExact `
            -Uri "$resultPrefix/DONE.json" -Destination (Join-Path $shardDir 'DONE.json') `
            -ProjectId $ProjectId
        $done = Get-Content -LiteralPath (Join-Path $shardDir 'DONE.json') -Raw | ConvertFrom-Json
        if ($done.schema -ne 'hu_m43_attempt06_spot_done_v1' -or
            $done.status -ne 'complete' -or [int]$done.shard -ne [int]$spec.shard -or
            [int]$done.root_index -ne [int]$spec.root_index -or
            [string]$done.manifest_sha256 -ne $manifestSha256 -or
            [string]$done.source_model_manifest_sha256 -ne [string]$manifest.source_model_manifest_sha256 -or
            [string]$done.source_native_manifest_sha256 -ne [string]$manifest.source_native_manifest_sha256 -or
            [string]$done.output_prefix -ne [string]$spec.output_prefix) {
            throw "Attempt06 DONE changed for shard $($spec.shard)"
        }
        $sources = @(
            "$resultPrefix/root.jsonl", "$resultPrefix/teacher.jsonl",
            "$resultPrefix/checkpoint.json", "$resultPrefix/heartbeat.json",
            "$resultPrefix/generator_summary.json", "$resultPrefix/run.log",
            "$resultPrefix/global_consumption_marker.json", "$resultPrefix/root_claim.json"
        )
        $copyJobs += [pscustomobject][ordered]@{
            label = "Attempt06 shard $($spec.shard) exact artifacts"
            sources = $sources
            destination = $shardDir
        }
    }
    [void](Invoke-M43A4ParallelExactGcsCopies `
        -CopyJobs $copyJobs -ProjectId $ProjectId -MaxParallel $MaxParallel `
        -TimeoutSecondsPerJob 1800)

    foreach ($spec in $specs) {
        $shardDir = Join-Path $shardsDir ([string]$spec.output_prefix)
        $auditPath = Join-Path $shardDir 'received_audit.json'
        $raw = Invoke-M43A6ReceivePython `
            -Arguments @(
                '-B', '-m', 'ofc_regular.hu_m43_attempt06_spot', 'validate-received-shard',
                '--shard-dir', $shardDir,
                '--schedule', (Join-Path $closureDir 'shards_manifest.jsonl'),
                '--shard', ([string]$spec.shard),
                '--consumption-marker', (Join-Path $closureDir 'audit_consumption_marker.json'),
                '--output', $auditPath
            ) -Label "validate received Attempt06 shard $($spec.shard)" -TimeoutSeconds 300
        $result = $raw.Trim() | ConvertFrom-Json
        if ($result.schema -ne 'hu_m43_attempt06_spot_received_shard_audit_v1' -or
            $result.status -ne 'pass_after_local_consumption_claim') {
            throw "Attempt06 shard audit failed: $($spec.shard)"
        }
    }

    $mergeRaw = Invoke-M43A6ReceivePython `
        -Arguments @(
            '-B', '-m', 'ofc_regular.hu_m43_attempt06_spot', 'merge-received',
            '--shards-root', $shardsDir,
            '--schedule', (Join-Path $closureDir 'shards_manifest.jsonl'),
            '--consumption-marker', (Join-Path $closureDir 'audit_consumption_marker.json'),
            '--output', (Join-Path $mergedDir 'teacher.jsonl'),
            '--receipt', (Join-Path $mergedDir 'merge_receipt.json')
        ) -Label 'merge exactly fifty Attempt06 shards' -TimeoutSeconds 600
    $merge = $mergeRaw.Trim() | ConvertFrom-Json
    if ($merge.schema -ne 'hu_m43_attempt06_spot_receive_merge_v1' -or
        $merge.status -ne 'merged_fifty_fresh_rows_without_fit_or_threshold_selection' -or
        [int]$merge.roots -ne 50 -or $merge.go_no_go_computed -ne $false) {
        throw 'Attempt06 receive merge boundary changed'
    }
    $receiveReceipt = [ordered]@{
        schema = 'hu_m43_attempt06_spot_receive_receipt_v1'
        status = 'verified_and_published_after_one_shot_claim'
        run_name = $RunName
        manifest_sha256 = $manifestSha256
        schedule_sha256 = [string]$manifest.schedule_sha256
        authorization_file_sha256 = $authorizationSha256
        consumption_marker_file_sha256 = Get-M43A4Sha256 $claimPath
        global_consumption_marker_sha256 = $globalMarkerSha256
        remote_audit_output_claim_uri = $remoteAuditOutputClaimUri
        remote_audit_output_claim_state = $remoteAuditOutputClaimState
        remote_audit_output_claim_sha256 = Get-M43A4Sha256 $claimPath
        merged_teacher_sha256 = Get-M43A4Sha256 (Join-Path $mergedDir 'teacher.jsonl')
        roots = 50
        shards = 50
        fit_performed = $false
        threshold_selected = $false
        go_no_go_computed = $false
        teacher_values_are_realized_match_ev = $false
        current_profile_mutated = $false
        runtime_policy_activated = $false
    }
    Write-M43A4Utf8CreateNew `
        -Path (Join-Path $mergedDir 'receive_receipt.json') `
        -Text (($receiveReceipt | ConvertTo-Json -Depth 6) + "`n")
    Publish-M43A6ReceiveDirectoryAtomic -Source $stagingRoot -Destination $OutputDir
}
catch {
    if ($stagingRoot -and (Test-Path -LiteralPath $stagingRoot)) {
        Remove-Item -LiteralPath $stagingRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
    throw
}
finally {
    Remove-Item -LiteralPath $temporaryGlobalMarker -Force -ErrorAction SilentlyContinue
}

[pscustomobject][ordered]@{
    schema = 'hu_m43_attempt06_spot_receive_result_v1'
    status = 'published_exact_fifty_root_audit'
    run_name = $RunName
    output_dir = $OutputDir
    merged_teacher = Join-Path $OutputDir 'merged/teacher.jsonl'
    consumption_marker = $claimPath
    roots = 50
    shards = 50
    fit_performed = $false
    threshold_selected = $false
    go_no_go_computed = $false
    teacher_values_are_realized_match_ev = $false
    current_profile_mutated = $false
    runtime_policy_activated = $false
} | ConvertTo-Json -Depth 6
