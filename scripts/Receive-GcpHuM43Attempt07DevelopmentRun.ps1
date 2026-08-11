param(
    [Parameter(Mandatory = $true)][string]$RunName,
    [Parameter(Mandatory = $true)][string]$SpotAuthorizationPath,
    [string]$ProjectId = 'ofc-solver-485418',
    [string]$Bucket = 'pokerhu-ofc-solver-485418-training',
    [string]$RunDir,
    [string]$OutputDir,
    [string]$ConsumptionClaimPath,
    [ValidateRange(1, 8)][int]$MaxParallel = 8,
    [switch]$ResumeClaim
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'HuM43Attempt04Spot.Common.ps1')

function Invoke-M43A7ReceivePython {
    param([string[]]$Arguments, [string]$Label, [int]$TimeoutSeconds = 600)
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

function Claim-M43A7RemoteOutputOnce {
    param([string]$LocalClaim, [string]$RemoteUri, [bool]$AllowExisting)
    $upload = Invoke-M43A4GcloudProcess `
        -Arguments @('storage','cp',$LocalClaim,$RemoteUri,'--project',$ProjectId,'--if-generation-match=0') `
        -TimeoutSeconds 300 -Label 'claim complete Attempt07 development output once'
    if (-not $upload.timed_out -and $upload.exit_code -eq 0) { return 'owner' }
    $existing = Join-Path ([IO.Path]::GetTempPath()) ('m43a7-output-claim-' + [guid]::NewGuid().ToString('N') + '.json')
    try {
        Copy-M43A4RemoteFileExact -Uri $RemoteUri -Destination $existing -ProjectId $ProjectId
        if ((Get-M43A4Sha256 $existing) -ne (Get-M43A4Sha256 $LocalClaim)) {
            throw 'remote Attempt07 output claim identity differs'
        }
        if (-not $AllowExisting) {
            throw 'remote Attempt07 output is already claimed; explicit ResumeClaim is required'
        }
        return 'existing'
    }
    finally { Remove-Item -LiteralPath $existing -Force -ErrorAction SilentlyContinue }
}

if ($RunName -notmatch '^[a-z0-9][a-z0-9-]{2,126}[a-z0-9]$') { throw 'unsafe RunName' }
$script:RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $RunDir) { $RunDir = Join-Path (Join-Path $script:RepoRoot 'outputs/gcp_runs') $RunName }
if (-not $OutputDir) {
    $OutputDir = Join-Path $script:RepoRoot "outputs/hu_joint_policy/m43_attempt07_development/$RunName"
}
if (-not $ConsumptionClaimPath) {
    $ConsumptionClaimPath = Join-Path $script:RepoRoot "outputs/hu_joint_policy/m43_attempt07_development/${RunName}_OUTPUT_CONSUMED.json"
}
$RunDir = Resolve-M43A4Path -Path $RunDir -Root $script:RepoRoot -Label 'Attempt07 run directory'
$OutputDir = Resolve-M43A4Path -Path $OutputDir -Root $script:RepoRoot -Label 'Attempt07 receive output'
$claimPath = Resolve-M43A4Path -Path $ConsumptionClaimPath -Root $script:RepoRoot -Label 'Attempt07 output claim'
$authorizationPath = Resolve-M43A4Path -Path $SpotAuthorizationPath -Root $script:RepoRoot -Label 'Attempt07 Spot authorization' -RequireFile
foreach ($binding in @(
    @($RunDir,'Attempt07 run directory'), @($OutputDir,'Attempt07 receive output'),
    @($claimPath,'Attempt07 output claim')
)) {
    Assert-M43A4UnderRoot -Path ([string]$binding[0]) -Root $script:RepoRoot -Label ([string]$binding[1])
}
if (Test-Path -LiteralPath $OutputDir) { throw 'Attempt07 receive output already exists' }
$outputFull = [IO.Path]::GetFullPath($OutputDir).TrimEnd([IO.Path]::DirectorySeparatorChar,[IO.Path]::AltDirectorySeparatorChar)
if ([IO.Path]::GetFullPath($claimPath).StartsWith(
    $outputFull + [IO.Path]::DirectorySeparatorChar,
    [StringComparison]::OrdinalIgnoreCase
)) { throw 'Attempt07 consumption claim must remain outside immutable receive output' }

$manifestPath = Resolve-M43A4Path -Path (Join-Path $RunDir 'manifest.json') -Root $script:RepoRoot -Label 'Attempt07 manifest' -RequireFile
$schedulePath = Resolve-M43A4Path -Path (Join-Path $RunDir 'shards_manifest.jsonl') -Root $script:RepoRoot -Label 'Attempt07 schedule' -RequireFile
$aggregatePath = Resolve-M43A4Path -Path (Join-Path $RunDir 'attempt07_preflight_aggregate.json') -Root $script:RepoRoot -Label 'Attempt07 aggregate' -RequireFile
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$manifestSha256 = Get-M43A4Sha256 $manifestPath
$authorizationSha256 = Get-M43A4Sha256 $authorizationPath
if ($manifest.schema -ne 'hu_m43_attempt07_development_spot_package_v1' -or
    $manifest.status -ne 'frozen_package_only_no_root_opened' -or
    [string]$manifest.run_name -ne $RunName -or [int]$manifest.total_shards -ne 100 -or
    [string]$manifest.schedule_sha256 -ne (Get-M43A4Sha256 $schedulePath) -or
    [string]$manifest.preflight_aggregate_sha256 -ne (Get-M43A4Sha256 $aggregatePath) -or
    $manifest.current_profile_mutated -ne $false -or $manifest.runtime_policy_activated -ne $false) {
    throw 'Attempt07 local receive closure changed'
}
$authAuditRaw = Invoke-M43A7ReceivePython `
    -Arguments @(
        '-B','-m','ofc_regular.hu_m43_attempt07_spot','validate-authorization',
        '--authorization',$authorizationPath,'--manifest',$manifestPath,
        '--preflight-aggregate',$aggregatePath
    ) -Label 'validate Attempt07 receive authorization'
$authAudit = $authAuditRaw.Trim() | ConvertFrom-Json
if ($authAudit.status -ne 'pass' -or $authAudit.all_gates_passed -ne $true) {
    throw 'Attempt07 receive authorization did not pass'
}

$specs = @([IO.File]::ReadLines($schedulePath) | Where-Object { $_.Trim() } | ForEach-Object { $_ | ConvertFrom-Json })
if ($specs.Count -ne 100) { throw 'Attempt07 receive requires exactly 100 schedule rows' }
$prefix = "gs://$Bucket/runs/$RunName"
$doneRoot = Join-Path ([IO.Path]::GetTempPath()) ('m43a7-done-only-' + [guid]::NewGuid().ToString('N'))
$stagingRoot = $null
New-Item -ItemType Directory -Path $doneRoot | Out-Null
try {
    # Phase 1 is deliberately DONE-only.  No result-content filename or URI is
    # formed until all 100 immutable operational markers pass and both local and
    # remote output claims exist.
    foreach ($spec in $specs) {
        $shard = [int]$spec.shard
        if ($shard -lt 0 -or $shard -ge 100 -or [int]$spec.root_index -ne $shard -or
            [string]$spec.output_prefix -ne ('shard_{0:D3}' -f $shard)) {
            throw "Attempt07 schedule ordering changed at shard $shard"
        }
        $doneUri = "$prefix/results/$($spec.output_prefix)/DONE.json"
        $donePath = Join-Path $doneRoot ('DONE-{0:D3}.json' -f $shard)
        Copy-M43A4RemoteFileExact -Uri $doneUri -Destination $donePath -ProjectId $ProjectId
    }

    $localClaimExisted = Test-Path -LiteralPath $claimPath
    $claimArgs = @(
        '-B','-m','ofc_regular.hu_m43_attempt07_spot','claim-complete-output',
        '--done-root',$doneRoot,'--schedule',$schedulePath,'--manifest',$manifestPath,
        '--authorization',$authorizationPath,'--preflight-aggregate',$aggregatePath,
        '--output',$claimPath
    )
    if ($localClaimExisted) {
        if (-not $ResumeClaim) { throw 'Attempt07 output claim exists; use ResumeClaim for this exact run only' }
        $claimArgs += '--resume-existing'
    }
    elseif ($ResumeClaim) { throw 'ResumeClaim requires an existing exact local claim' }
    $claimRaw = Invoke-M43A7ReceivePython -Arguments $claimArgs -Label 'claim all 100 Attempt07 DONE records'
    $claim = $claimRaw.Trim() | ConvertFrom-Json
    if ($claim.schema -ne 'hu_m43_attempt07_development_output_consumption_v1' -or
        $claim.status -ne 'claimed_after_all_done_before_any_teacher_read' -or
        $claim.all_done_markers_verified -ne $true -or [int]$claim.expected_shards -ne 100 -or
        $claim.result_objects_addressed_when_claimed -ne $false -or
        $claim.selector_executed -ne $false -or $claim.current_profile_mutated -ne $false) {
        throw 'Attempt07 output consumption claim changed'
    }
    $remoteClaimUri = "$prefix/development_output_boundary/CONSUMED.json"
    $remoteClaimState = Claim-M43A7RemoteOutputOnce `
        -LocalClaim $claimPath -RemoteUri $remoteClaimUri `
        -AllowExisting ($localClaimExisted -and [bool]$ResumeClaim)

    # Phase 2 begins only after complete DONE verification and the irreversible
    # local+remote claims.  These are the first result-content paths in this file.
    $contentName = 'teacher.jsonl'
    $stagingRoot = $OutputDir + '.staging-' + [guid]::NewGuid().ToString('N')
    $closureDir = Join-Path $stagingRoot 'closure'
    $shardsDir = Join-Path $stagingRoot 'shards'
    $mergedDir = Join-Path $stagingRoot 'merged'
    New-Item -ItemType Directory -Path $closureDir,$shardsDir,$mergedDir | Out-Null
    foreach ($name in @(
        'manifest.json','shards_manifest.jsonl','source_closure_manifest.json',
        'hu_joint_policy_m43_attempt07.json','hu_joint_policy_m43_attempt07_status.json',
        'hu_joint_policy_m43_attempt07_preflight.json','attempt07_preflight_aggregate.json',
        'startup_hu_m43_attempt07_development.sh'
    )) {
        $source = Join-Path $RunDir $name
        if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { throw "Attempt07 receive closure missing: $source" }
        Copy-Item -LiteralPath $source -Destination (Join-Path $closureDir $name)
    }
    Copy-Item -LiteralPath (Join-Path $RunDir 'preflight_closure') -Destination (Join-Path $closureDir 'preflight_closure') -Recurse
    Copy-Item -LiteralPath (Join-Path $RunDir 'package_src/source_model_manifest.json') -Destination (Join-Path $closureDir 'source_model_manifest.json')
    Copy-Item -LiteralPath (Join-Path $RunDir 'package_src/source_native_manifest.json') -Destination (Join-Path $closureDir 'source_native_manifest.json')
    Copy-Item -LiteralPath $authorizationPath -Destination (Join-Path $closureDir 'spot_authorization.json')
    Copy-Item -LiteralPath $claimPath -Destination (Join-Path $closureDir 'output_consumption_claim.json')

    $copyJobs = @()
    foreach ($spec in $specs) {
        $shard = [int]$spec.shard
        $shardDir = Join-Path $shardsDir ([string]$spec.output_prefix)
        New-Item -ItemType Directory -Path $shardDir | Out-Null
        Copy-Item -LiteralPath (Join-Path $doneRoot ('DONE-{0:D3}.json' -f $shard)) -Destination (Join-Path $shardDir 'DONE.json')
        $contentPrefix = "$prefix/results/$($spec.output_prefix)"
        $sources = @(
            "$contentPrefix/$contentName", "$contentPrefix/checkpoint.json",
            "$contentPrefix/heartbeat.json", "$contentPrefix/generator_summary.json",
            "$contentPrefix/run.log", "$contentPrefix/authorization.json",
            "$contentPrefix/global_claim.json", "$contentPrefix/root_claim.json"
        )
        $copyJobs += [pscustomobject][ordered]@{
            label = "Attempt07 shard $shard exact content"
            sources = $sources
            destination = $shardDir
        }
    }
    [void](Invoke-M43A4ParallelExactGcsCopies `
        -CopyJobs $copyJobs -ProjectId $ProjectId -MaxParallel $MaxParallel `
        -TimeoutSecondsPerJob 1800)

    foreach ($spec in $specs) {
        $shard = [int]$spec.shard
        $shardDir = Join-Path $shardsDir ([string]$spec.output_prefix)
        $auditPath = Join-Path $shardDir 'received_audit.json'
        $auditRaw = Invoke-M43A7ReceivePython `
            -Arguments @(
                '-B','-m','ofc_regular.hu_m43_attempt07_spot','validate-received-shard',
                '--shard-dir',$shardDir,'--schedule',(Join-Path $closureDir 'shards_manifest.jsonl'),
                '--manifest',(Join-Path $closureDir 'manifest.json'),'--output',$auditPath
            ) -Label "validate Attempt07 received shard $shard" -TimeoutSeconds 900
        $audit = $auditRaw.Trim() | ConvertFrom-Json
        if ($audit.schema -ne 'hu_m43_attempt07_development_received_shard_v1' -or
            $audit.status -ne 'pass' -or [int]$audit.shard -ne $shard) {
            throw "Attempt07 shard audit failed: $shard"
        }
    }

    $mergeRaw = Invoke-M43A7ReceivePython `
        -Arguments @(
            '-B','-m','ofc_regular.hu_m43_attempt07_spot','merge-received',
            '--received-root',$shardsDir,'--schedule',(Join-Path $closureDir 'shards_manifest.jsonl'),
            '--manifest',(Join-Path $closureDir 'manifest.json'),
            '--consumption-claim',(Join-Path $closureDir 'output_consumption_claim.json'),
            '--output',(Join-Path $mergedDir $contentName),
            '--receipt',(Join-Path $mergedDir 'merge_receipt.json')
        ) -Label 'merge exactly 100 audited Attempt07 rows' -TimeoutSeconds 1200
    $merge = $mergeRaw.Trim() | ConvertFrom-Json
    if ($merge.schema -ne 'hu_m43_attempt07_development_receive_merge_v1' -or
        $merge.status -ne 'complete_no_selection' -or [int]$merge.roots -ne 100 -or
        $merge.selector_executed -ne $false -or $merge.current_profile_mutated -ne $false) {
        throw 'Attempt07 development merge boundary changed'
    }
    $receipt = [ordered]@{
        schema = 'hu_m43_attempt07_development_receive_receipt_v1'
        status = 'verified_exact_100_without_selection'
        run_name = $RunName
        manifest_sha256 = $manifestSha256
        authorization_sha256 = $authorizationSha256
        consumption_claim_sha256 = Get-M43A4Sha256 $claimPath
        remote_claim_uri = $remoteClaimUri
        remote_claim_state = $remoteClaimState
        merged_sha256 = Get-M43A4Sha256 (Join-Path $mergedDir $contentName)
        roots = 100
        selector_executed = $false
        selector_command_required = 'python -B -m ofc_regular.select_hu_m43_attempt07_development_arm'
        fit_performed = $false
        threshold_selected = $false
        current_profile_mutated = $false
        runtime_policy_activated = $false
    }
    Write-M43A4Utf8CreateNew -Path (Join-Path $mergedDir 'receive_receipt.json') -Text (($receipt | ConvertTo-Json -Depth 8) + "`n")
    [IO.Directory]::Move([IO.Path]::GetFullPath($stagingRoot),[IO.Path]::GetFullPath($OutputDir))
    $stagingRoot = $null
}
catch {
    if ($stagingRoot -and (Test-Path -LiteralPath $stagingRoot)) {
        Remove-Item -LiteralPath $stagingRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
    throw
}
finally { Remove-Item -LiteralPath $doneRoot -Recurse -Force -ErrorAction SilentlyContinue }

[pscustomobject][ordered]@{
    schema = 'hu_m43_attempt07_development_receive_result_v1'
    status = 'published_exact_100_without_selection'
    run_name = $RunName
    output_dir = $OutputDir
    consumption_claim = $claimPath
    roots = 100
    selector_executed = $false
    selector_command = "`$env:PYTHONPATH='src'; python -B -m ofc_regular.select_hu_m43_attempt07_development_arm --input '$OutputDir/merged/teacher.jsonl' --plan 'configs/hu_joint_policy_m43_attempt07.json' --output '<new-selection.json>'"
    current_profile_mutated = $false
    runtime_policy_activated = $false
} | ConvertTo-Json -Depth 8
