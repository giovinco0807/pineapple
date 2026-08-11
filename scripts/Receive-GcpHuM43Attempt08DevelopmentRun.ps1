param(
    [Parameter(Mandatory = $true)][string]$RunName,
    [string]$ProjectId = 'ofc-solver-485418',
    [string]$Bucket = 'pokerhu-ofc-solver-485418-training',
    [string]$Zone = 'asia-northeast1-b',
    [string]$RunDir,
    [ValidateRange(1, 8)][int]$MaxParallel = 8,
    [switch]$ResumeSelector
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if ($RunName -notmatch '^[a-z0-9][a-z0-9-]{2,126}[a-z0-9]$') {
    throw 'Unsafe Attempt08 RunName'
}
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $RunDir) { $RunDir = Join-Path $repoRoot "outputs/gcp_runs/$RunName" }
$RunDir = [IO.Path]::GetFullPath($RunDir)
$expectedRunDir = [IO.Path]::GetFullPath((Join-Path $repoRoot "outputs/gcp_runs/$RunName"))
if (-not [string]::Equals($RunDir, $expectedRunDir, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Attempt08 receive RunDir must be the exact repository run directory'
}
$frozenScript = Join-Path $RunDir 'package_src/scripts/Receive-GcpHuM43Attempt08DevelopmentRun.ps1'
$frozenCommon = Join-Path $RunDir 'package_src/scripts/HuM43Attempt08Spot.Common.ps1'
foreach ($path in @($frozenScript,$frozenCommon,(Join-Path $RunDir 'package_src/scripts/HuM43Attempt04Spot.Common.ps1'))) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Attempt08 frozen receive source is missing: $path" }
}
if ((Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash -ne
    (Get-FileHash -LiteralPath $frozenScript -Algorithm SHA256).Hash) {
    throw 'Attempt08 receive script differs from its frozen packaged copy'
}
. $frozenCommon
$manifestPath = Join-Path $RunDir 'manifest.json'
$authorizationPath = Join-Path $RunDir 'launch_authorization.json'
$schedulePath = Join-Path $RunDir 'shards_manifest.jsonl'
foreach ($path in @($manifestPath,$authorizationPath,$schedulePath)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Attempt08 receive package file is missing: $path"
    }
}

$env:PYTHONDONTWRITEBYTECODE = '1'
$frozenModuleRoot = Join-Path $RunDir 'package_src'
[void](Invoke-M43A8Python -RepoRoot $frozenModuleRoot -Arguments @(
    '-B','-m','ofc_regular.hu_m43_attempt08_spot','validate-launch',
    '--run-dir',$RunDir,'--authorization',$authorizationPath
) -Label 'validate frozen Attempt08 receive package' -TimeoutSeconds 3600)

$schedule = @(Get-Content -LiteralPath $schedulePath | Where-Object { $_ } | ForEach-Object { $_ | ConvertFrom-Json })
if ($schedule.Count -ne 200) { throw 'Attempt08 receive requires exactly 200 schedule rows' }
for ($index = 0; $index -lt 200; $index++) {
    $spec = $schedule[$index]
    if ([int]$spec.shard -ne $index -or [int]$spec.root_index -ne $index -or
        [string]$spec.output_prefix -ne ('shard_{0:d3}' -f $index)) {
        throw "Attempt08 receive schedule mapping changed at shard $index"
    }
}

$base = [IO.Path]::GetFullPath((Join-Path $repoRoot "outputs/hu_joint_policy/m43_attempt08_development/$RunName"))
$developmentRoot = [IO.Path]::GetFullPath((Join-Path $repoRoot 'outputs/hu_joint_policy/m43_attempt08_development'))
if (-not $base.StartsWith($developmentRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Attempt08 receive output escaped its canonical root'
}
$resumeSelectorOnly = Test-Path -LiteralPath $base
if ($resumeSelectorOnly -and -not $ResumeSelector) {
    throw 'Attempt08 canonical receive output exists; use ResumeSelector for the exact post-merge lifecycle only'
}
if (-not $resumeSelectorOnly -and $ResumeSelector) {
    throw 'ResumeSelector requires an existing canonical Attempt08 receive output'
}
$staging = $base + '.receive-building'
if (Test-Path -LiteralPath $staging) { throw 'Stale Attempt08 receive staging directory exists' }
$prefix = "gs://$Bucket/runs/$RunName"

function Complete-Attempt08Selector {
    $localClaimPath = Join-Path $base 'claims/CONSUMED.json'
    $remoteClaimPath = Join-Path $base 'claims/REMOTE_CONSUMED.json'
    $mergedPath = Join-Path $base 'merged/teacher.jsonl'
    $mergeReceiptPath = Join-Path $base 'merged/merge_receipt.json'
    [void](Invoke-M43A8Python -RepoRoot $frozenModuleRoot -Arguments @(
        '-B','-m','ofc_regular.hu_m43_attempt08_spot','validate-claims',
        '--run-dir',$RunDir,'--authorization',$authorizationPath,
        '--local-claim',$localClaimPath,'--remote-claim',$remoteClaimPath
    ) -Label 'reopen Attempt08 canonical consumption claims' -TimeoutSeconds 3600)
    [void](Invoke-M43A8Python -RepoRoot $frozenModuleRoot -Arguments @(
        '-B','-m','ofc_regular.hu_m43_attempt08_spot','validate-merged-receive',
        '--run-dir',$RunDir,'--authorization',$authorizationPath,
        '--local-claim',$localClaimPath,'--remote-claim',$remoteClaimPath
    ) -Label 'reopen exact Attempt08 200-shard merge before selector lifecycle' -TimeoutSeconds 3600)
    $selectorRootPath = Join-Path $base 'selector'
    New-Item -ItemType Directory -Path $selectorRootPath -Force | Out-Null
    $selectorClaimPath = Join-Path $selectorRootPath 'CLAIM.json'
    $remoteSelectorClaimPath = Join-Path $selectorRootPath 'REMOTE_CLAIM.json'
    $executionPath = Join-Path $selectorRootPath 'EXECUTION_STARTED.json'
    $decisionPath = Join-Path $selectorRootPath 'decision.json'
    $decisionReceiptPath = Join-Path $selectorRootPath 'decision_receipt.json'
    $executionExists = Test-Path -LiteralPath $executionPath -PathType Leaf
    $decisionExists = Test-Path -LiteralPath $decisionPath -PathType Leaf
    $receiptExists = Test-Path -LiteralPath $decisionReceiptPath -PathType Leaf
    if ($executionExists -or $decisionExists -or $receiptExists) {
        if (-not ($executionExists -and $decisionExists -and $receiptExists)) {
            throw 'Attempt08 selector began but did not atomically finish; a separate recovery audit is required and automatic gate reevaluation is forbidden'
        }
        $completedRaw = Invoke-M43A8Python -RepoRoot $frozenModuleRoot -Arguments @(
            '-B','-m','ofc_regular.hu_m43_attempt08_spot','validate-selector-completion',
            '--run-dir',$RunDir
        ) -Label 'validate existing completed Attempt08 selector without reevaluation' -TimeoutSeconds 3600
        return ($completedRaw.Trim() | ConvertFrom-Json)
    }
    if (-not (Test-Path -LiteralPath $selectorClaimPath -PathType Leaf)) {
        [void](Invoke-M43A8Python -RepoRoot $frozenModuleRoot -Arguments @(
            '-B','-m','ofc_regular.hu_m43_attempt08_spot','selector-claim',
            '--run-dir',$RunDir,'--authorization',$authorizationPath,
            '--local-consumption-claim',$localClaimPath,'--remote-consumption-claim',$remoteClaimPath,
            '--merged',$mergedPath,'--merge-receipt',$mergeReceiptPath,'--output',$selectorClaimPath
        ) -Label 'claim sole Attempt08 selector execution' -TimeoutSeconds 3600)
    }
    $remoteSelectorUri = "$prefix/claims/development_selector.json"
    [void](Publish-M43A8ImmutableObject -Source $selectorClaimPath -Uri $remoteSelectorUri -ProjectId $ProjectId)
    if (Test-Path -LiteralPath $remoteSelectorClaimPath -PathType Leaf) {
        if ((Get-M43A4Sha256 $remoteSelectorClaimPath) -ne (Get-M43A4Sha256 $selectorClaimPath)) {
            throw 'Attempt08 downloaded remote selector claim differs from the local claim'
        }
    }
    else {
        Copy-M43A4RemoteFileExact -Uri $remoteSelectorUri -Destination $remoteSelectorClaimPath `
            -ProjectId $ProjectId -ExpectedSha256 (Get-M43A4Sha256 $selectorClaimPath)
    }
    $selectionRaw = Invoke-M43A8Python -RepoRoot $frozenModuleRoot -Arguments @(
        '-B','-m','ofc_regular.hu_m43_attempt08_spot','select-once',
        '--run-dir',$RunDir,'--selector-claim',$selectorClaimPath,
        '--remote-selector-claim',$remoteSelectorClaimPath,
        '--output',$decisionPath,'--receipt',$decisionReceiptPath
    ) -Label 'execute frozen Attempt08 selector exactly once' -TimeoutSeconds 3600
    return ($selectionRaw.Trim() | ConvertFrom-Json)
}

function Write-Attempt08ReceiveResult {
    param([Parameter(Mandatory = $true)]$Selection)
    if ($Selection.status -ne 'single_frozen_gate_evaluation_complete' -or
        [int]$Selection.gate_evaluation_count -ne 1 -or $Selection.selector_executed -ne $true -or
        $Selection.current_profile_mutated -ne $false -or $Selection.runtime_policy_activated -ne $false) {
        throw 'Attempt08 selector execution boundary changed'
    }
    [pscustomobject]@{
        schema = 'hu_m43_attempt08_development_receive_result_v1'
        status = 'exact_200_received_and_selector_executed_once'
        run_name = $RunName
        output_dir = $base
        merged = (Join-Path $base 'merged/teacher.jsonl')
        decision = (Join-Path $base 'selector/decision.json')
        decision_receipt = (Join-Path $base 'selector/decision_receipt.json')
        roots = 200
        gate_evaluation_count = 1
        selector_executed = $true
        future_audit_authorized = $false
        current_profile_mutated = $false
        runtime_policy_activated = $false
    } | ConvertTo-Json -Depth 8
}

if ($resumeSelectorOnly) {
    $resumedSelection = Complete-Attempt08Selector
    Write-Attempt08ReceiveResult -Selection $resumedSelection
    return
}

$doneRoot = Join-Path $staging 'done'
$receivedRoot = Join-Path $staging 'shards'
$auditRoot = Join-Path $staging 'audits'
$mergedRoot = Join-Path $staging 'merged'
$claimsRoot = Join-Path $staging 'claims'
New-Item -ItemType Directory -Path $doneRoot,$receivedRoot,$auditRoot,$mergedRoot,$claimsRoot | Out-Null
$remoteConsumptionUri = "$prefix/claims/development_output_consumption.json"
$localClaim = Join-Path $claimsRoot 'CONSUMED.json'
$remoteClaim = Join-Path $claimsRoot 'REMOTE_CONSUMED.json'
$moved = $false

try {
    # Phase 1: address only the exact DONE object for every scheduled shard.
    # Result-content filenames are intentionally introduced only after both
    # byte-identical consumption claims have been established.
    foreach ($spec in $schedule) {
        $shard = [int]$spec.shard
        Copy-M43A4RemoteFileExact `
            -Uri "$prefix/results/$($spec.output_prefix)/DONE.json" `
            -Destination (Join-Path $doneRoot ('DONE-{0:d3}.json' -f $shard)) `
            -ProjectId $ProjectId
    }
    [void](Invoke-M43A8Python -RepoRoot $frozenModuleRoot -Arguments @(
        '-B','-m','ofc_regular.hu_m43_attempt08_spot','validate-done-set',
        '--done-root',$doneRoot,'--run-dir',$RunDir,'--authorization',$authorizationPath
    ) -Label 'validate exact Attempt08 200-DONE set' -TimeoutSeconds 3600)
    $claimRaw = Invoke-M43A8Python -RepoRoot $frozenModuleRoot -Arguments @(
        '-B','-m','ofc_regular.hu_m43_attempt08_spot','claim',
        '--done-root',$doneRoot,'--run-dir',$RunDir,'--authorization',$authorizationPath,
        '--output',$localClaim
    ) -Label 'claim complete Attempt08 output before content read' -TimeoutSeconds 3600
    $claim = $claimRaw.Trim() | ConvertFrom-Json
    if ($claim.status -ne 'claimed_after_all_done_before_any_teacher_read' -or
        [int]$claim.expected_shards -ne 200 -or $claim.all_done_markers_verified -ne $true -or
        $claim.result_objects_addressed_when_claimed -ne $false -or
        $claim.remote_claim_required_before_content_read -ne $true -or
        $claim.selector_executed -ne $false -or $claim.current_profile_mutated -ne $false) {
        throw 'Attempt08 local consumption boundary changed'
    }
    [void](Publish-M43A8ImmutableObject -Source $localClaim -Uri $remoteConsumptionUri -ProjectId $ProjectId)
    Copy-M43A4RemoteFileExact -Uri $remoteConsumptionUri -Destination $remoteClaim -ProjectId $ProjectId `
        -ExpectedSha256 (Get-M43A4Sha256 $localClaim)
    [void](Invoke-M43A8Python -RepoRoot $frozenModuleRoot -Arguments @(
        '-B','-m','ofc_regular.hu_m43_attempt08_spot','validate-claims',
        '--run-dir',$RunDir,'--authorization',$authorizationPath,
        '--local-claim',$localClaim,'--remote-claim',$remoteClaim
    ) -Label 'validate byte-identical local and remote Attempt08 consumption claims' -TimeoutSeconds 3600)

    # Phase 2: the first result-content addressing happens only after both
    # claims exist and have been reopened byte-for-byte.
    $contentNames = @(
        'teacher.jsonl','checkpoint.json','heartbeat.json','generator_summary.json',
        'run.log','boot_image_evidence.json','time.txt','resume_commit.json',
        'global_claim.json','root_claim.json'
    )
    $copyJobs = [Collections.Generic.List[object]]::new()
    foreach ($spec in $schedule) {
        $shard = [int]$spec.shard
        $directory = Join-Path $receivedRoot ([string]$spec.output_prefix)
        New-Item -ItemType Directory -Path $directory | Out-Null
        Copy-Item -LiteralPath (Join-Path $doneRoot ('DONE-{0:d3}.json' -f $shard)) `
            -Destination (Join-Path $directory 'DONE.json')
        $sources = @($contentNames | ForEach-Object { "$prefix/results/$($spec.output_prefix)/$_" })
        $copyJobs.Add([pscustomobject]@{
            label = "download exact Attempt08 content for shard $shard"
            sources = $sources
            destination = $directory
        })
    }
    [void](Invoke-M43A4ParallelExactGcsCopies -CopyJobs @($copyJobs) `
        -ProjectId $ProjectId -MaxParallel $MaxParallel -TimeoutSecondsPerJob 1800)

    foreach ($spec in $schedule) {
        $shard = [int]$spec.shard
        $directory = Join-Path $receivedRoot ([string]$spec.output_prefix)
        $boot = Get-Content -LiteralPath (Join-Path $directory 'boot_image_evidence.json') -Raw | ConvertFrom-Json
        $diskResult = Invoke-M43A4GcloudProcess -Arguments @(
            'compute','disks','describe',([string]$boot.disk_name),'--zone',$Zone,
            '--project',$ProjectId,'--format=json'
        ) -TimeoutSeconds 60 -Label "recheck Attempt08 boot disk $shard"
        if (-not $diskResult.timed_out -and $diskResult.exit_code -eq 0) {
            $disk = $diskResult.stdout | ConvertFrom-Json
            if ([string]$disk.sourceImageId -ne '1449487925682397051' -or
                -not ([string]$disk.sourceImage).EndsWith('/projects/debian-cloud/global/images/debian-12-bookworm-v20260609')) {
                throw "Attempt08 live boot disk image changed at shard $shard"
            }
        }
        else {
            $failure = ([string]$diskResult.stdout) + ([string]$diskResult.stderr)
            if ($diskResult.timed_out -or $failure -notmatch '(?i)not found|was not found|404') {
                throw "Unable to verify Attempt08 boot disk state at shard $shard"
            }
        }
        $auditPath = Join-Path $auditRoot ('audit-{0:d3}.json' -f $shard)
        $auditRaw = Invoke-M43A8Python -RepoRoot $frozenModuleRoot -Arguments @(
            '-B','-m','ofc_regular.hu_m43_attempt08_spot','audit-received',
            '--directory',$directory,'--shard',"$shard",'--done-root',$doneRoot,
            '--run-dir',$RunDir,'--authorization',$authorizationPath,
            '--local-claim',$localClaim,'--remote-claim',$remoteClaim,'--output',$auditPath
        ) -Label "audit complete Attempt08 received shard $shard" -TimeoutSeconds 3600
        $audit = $auditRaw.Trim() | ConvertFrom-Json
        if ($audit.status -ne 'verified_after_local_and_remote_consumption_claims' -or
            [int]$audit.shard -ne $shard -or $audit.selector_executed -ne $false -or
            $audit.current_profile_mutated -ne $false) {
            throw "Attempt08 received audit boundary changed at shard $shard"
        }
    }

    $merged = Join-Path $mergedRoot 'teacher.jsonl'
    $mergeReceipt = Join-Path $mergedRoot 'merge_receipt.json'
    $mergeRaw = Invoke-M43A8Python -RepoRoot $frozenModuleRoot -Arguments @(
        '-B','-m','ofc_regular.hu_m43_attempt08_spot','merge-received',
        '--received-root',$receivedRoot,'--audit-root',$auditRoot,'--done-root',$doneRoot,
        '--run-dir',$RunDir,'--authorization',$authorizationPath,
        '--local-claim',$localClaim,'--remote-claim',$remoteClaim,
        '--output',$merged,'--receipt',$mergeReceipt
    ) -Label 'merge exact 200 audited Attempt08 rows' -TimeoutSeconds 3600
    $merge = $mergeRaw.Trim() | ConvertFrom-Json
    if ($merge.status -ne 'complete_without_selection' -or [int]$merge.roots -ne 200 -or
        $merge.selector_executed -ne $false -or $merge.selector_must_execute_exactly_once -ne $true -or
        $merge.current_profile_mutated -ne $false) {
        throw 'Attempt08 merge boundary changed'
    }

    [IO.Directory]::Move([IO.Path]::GetFullPath($staging), $base)
    $moved = $true
    $selection = Complete-Attempt08Selector
    Write-Attempt08ReceiveResult -Selection $selection
}
catch {
    if (-not $moved -and (Test-Path -LiteralPath $staging)) {
        $resolvedStaging = [IO.Path]::GetFullPath($staging)
        if (-not [string]::Equals($resolvedStaging, $base + '.receive-building', [StringComparison]::OrdinalIgnoreCase)) {
            throw 'Attempt08 receive cleanup target changed'
        }
        Remove-Item -LiteralPath $resolvedStaging -Recurse -Force
    }
    throw
}
