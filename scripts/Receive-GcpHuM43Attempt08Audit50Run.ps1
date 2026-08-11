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
$frozenScript = Join-Path $RunDir 'overlay_src/scripts/Receive-GcpHuM43Attempt08Audit50Run.ps1'
$frozenCommon = Join-Path $RunDir 'overlay_src/scripts/HuM43Attempt08Audit50Spot.Common.ps1'
if ((Get-FileHash $PSCommandPath -Algorithm SHA256).Hash -ne (Get-FileHash $frozenScript -Algorithm SHA256).Hash) { throw 'Attempt08 audit50 Receive script differs from frozen overlay' }
. $frozenCommon
$module = Join-Path $RunDir 'overlay_src/src/ofc_regular/hu_m43_attempt08_audit50_spot.py'
$authorization = Join-Path $RunDir 'launch_authorization.json'
[void](Invoke-M43A8Audit50Python -ModuleFile $module -Arguments @('validate-launch','--run-dir',$RunDir,'--development-run-dir',$DevelopmentRunDir,'--authorization',$authorization) -Label 'validate audit50 receive chain' -TimeoutSeconds 3600)

$base = Join-Path $repoRoot "outputs/hu_joint_policy/m43_attempt08_audit50/$RunName"
$staging = "$base.receiving"
$prefix = "gs://$Bucket/runs/$RunName"

function Assert-M43A8Audit50SafePreSelectorState {
    param([Parameter(Mandatory = $true)][string]$Root)
    foreach ($relative in @('selector/EXECUTION_STARTED.json','selector/decision.json')) {
        if (Test-Path -LiteralPath (Join-Path $Root $relative) -PathType Leaf) {
            throw "Attempt08 audit50 selector execution already began without a receipt; automatic gate reevaluation is forbidden: $relative"
        }
    }
}

function Sync-M43A8Audit50ImmutableRemoteCopy {
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [Parameter(Mandatory = $true)][string]$Destination
    )
    if (-not (Test-Path -LiteralPath $Destination -PathType Leaf)) {
        Copy-M43A4RemoteFileExact -Uri $Uri -Destination $Destination -ProjectId $ProjectId
        return
    }
    $temporary = Join-Path ([IO.Path]::GetTempPath()) ('m43a8-audit50-remote-' + [guid]::NewGuid().ToString('N'))
    try {
        Copy-M43A4RemoteFileExact -Uri $Uri -Destination $temporary -ProjectId $ProjectId
        if ((Get-M43A4Sha256 $temporary) -ne (Get-M43A4Sha256 $Destination)) {
            throw "Attempt08 audit50 cached immutable remote copy differs: $Uri"
        }
    }
    finally { Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue }
}

$resumePreSelector = $false
if (Test-Path $base) {
    $receipt = Join-Path $base 'selector/decision_receipt.json'
    if (Test-Path $receipt -PathType Leaf) {
        $raw = Invoke-M43A8Audit50Python -ModuleFile $module -Arguments @('validate-selector','--run-dir',$base) -Label 'validate completed audit50 selector without reevaluation'
        [pscustomobject]@{schema='hu_m43_attempt08_audit50_receive_result_v1';status='already_received_selector_validated_without_reevaluation';run_name=$RunName;selector=($raw.Trim() | ConvertFrom-Json);current_profile_mutated=$false;runtime_policy_activated=$false} | ConvertTo-Json -Depth 10
        return
    }
    Assert-M43A8Audit50SafePreSelectorState -Root $base
    $resumePreSelector = $true
}
if (Test-Path -LiteralPath $staging) {
    if ($resumePreSelector) { throw 'Attempt08 audit50 canonical receive and .receiving staging both exist' }
    Assert-M43A8Audit50SafePreSelectorState -Root $staging
    $expectedStaging = [IO.Path]::GetFullPath("$base.receiving")
    if ([IO.Path]::GetFullPath($staging) -ne $expectedStaging) { throw 'Attempt08 audit50 staging path changed' }
    # Phase 1/2 is deterministic from immutable remote objects.  A crash-safe
    # restart discards only the noncanonical pre-selector staging tree.
    Remove-Item -LiteralPath $staging -Recurse -Force
}

if (-not $resumePreSelector) {
    $doneRoot = Join-Path $staging 'done_only'; $receivedRoot = Join-Path $staging 'received'; $auditRoot = Join-Path $staging 'audits'; $mergedRoot = Join-Path $staging 'merged'; $claimsRoot = Join-Path $staging 'claims'
    New-Item -ItemType Directory -Path $doneRoot,$receivedRoot,$auditRoot,$mergedRoot,$claimsRoot | Out-Null

    # Phase 1: address only DONE. Package/source/model hashes and the frozen
    # schedule were already validated; no audit result object is read here.
    for ($shard = 0; $shard -lt 50; $shard++) {
        Copy-M43A4RemoteFileExact -Uri "$prefix/results/shard_$('{0:d3}' -f $shard)/DONE.json" -Destination (Join-Path $doneRoot ('DONE-{0:d3}.json' -f $shard)) -ProjectId $ProjectId
    }
    [void](Invoke-M43A8Audit50Python -ModuleFile $module -Arguments @('validate-done-set','--run-dir',$RunDir,'--authorization',$authorization,'--done-root',$doneRoot) -Label 'validate exact audit50 DONE set')
    $localClaim = Join-Path $claimsRoot 'CONSUMED.json'; $remoteClaim = Join-Path $claimsRoot 'REMOTE_CONSUMED.json'
    [void](Invoke-M43A8Audit50Python -ModuleFile $module -Arguments @('claim','--run-dir',$RunDir,'--authorization',$authorization,'--done-root',$doneRoot,'--output',$localClaim) -Label 'claim audit50 result content before read')
    $remoteClaimUri = "$prefix/claims/audit50_output_consumption.json"
    [void](Publish-M43A8Audit50ImmutableObject -Source $localClaim -Uri $remoteClaimUri -ProjectId $ProjectId)
    Sync-M43A8Audit50ImmutableRemoteCopy -Uri $remoteClaimUri -Destination $remoteClaim
    [void](Invoke-M43A8Audit50Python -ModuleFile $module -Arguments @('validate-claims','--run-dir',$RunDir,'--authorization',$authorization,'--local-claim',$localClaim,'--remote-claim',$remoteClaim) -Label 'validate byte-identical audit50 consumption claims')

    # Phase 2: after both claims, download and fully audit every immutable bundle.
    $names = @('teacher.jsonl','checkpoint.json','heartbeat.json','generator_summary.json','run.log','time.txt','global_claim.json','root_claim.json','boot_image_evidence.json','resume_commit.json','DONE.json')
    for ($shard = 0; $shard -lt 50; $shard++) {
        $directory = Join-Path $receivedRoot ('shard_{0:d3}' -f $shard)
        New-Item -ItemType Directory -Path $directory | Out-Null
        foreach ($name in $names) { Copy-M43A4RemoteFileExact -Uri "$prefix/results/shard_$('{0:d3}' -f $shard)/$name" -Destination (Join-Path $directory $name) -ProjectId $ProjectId }
        $auditPath = Join-Path $auditRoot ('audit_{0:d3}.json' -f $shard)
        [void](Invoke-M43A8Audit50Python -ModuleFile $module -Arguments @('audit-received','--directory',$directory,'--shard',[string]$shard,'--run-dir',$RunDir,'--development-run-dir',$DevelopmentRunDir,'--authorization',$authorization,'--local-claim',$localClaim,'--remote-claim',$remoteClaim,'--output',$auditPath) -Label "audit received audit50 shard $shard" -TimeoutSeconds 3600)
    }
    $merged = Join-Path $mergedRoot 'teacher.jsonl'; $mergeReceipt = Join-Path $mergedRoot 'merge_receipt.json'
    [void](Invoke-M43A8Audit50Python -ModuleFile $module -Arguments @('merge-received','--received-root',$receivedRoot,'--audit-root',$auditRoot,'--run-dir',$RunDir,'--authorization',$authorization,'--local-claim',$localClaim,'--remote-claim',$remoteClaim,'--output',$merged,'--receipt',$mergeReceipt) -Label 'merge exact 50 audit50 rows')
    [void](Invoke-M43A8Audit50Python -ModuleFile $module -Arguments @('validate-merged','--run-dir',$RunDir,'--authorization',$authorization,'--local-claim',$localClaim,'--remote-claim',$remoteClaim,'--merged',$merged,'--receipt',$mergeReceipt) -Label 'validate audit50 canonical merge')
    Move-Item -LiteralPath $staging -Destination $base
}
else {
    # Crash recovery after the atomic Move but before selector execution.
    $doneRoot = Join-Path $base 'done_only'; $localClaim = Join-Path $base 'claims/CONSUMED.json'; $remoteClaim = Join-Path $base 'claims/REMOTE_CONSUMED.json'; $merged = Join-Path $base 'merged/teacher.jsonl'; $mergeReceipt = Join-Path $base 'merged/merge_receipt.json'
    foreach ($required in @($localClaim,$remoteClaim,$merged,$mergeReceipt)) { if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "Attempt08 audit50 pre-selector recovery artifact missing: $required" } }
    [void](Invoke-M43A8Audit50Python -ModuleFile $module -Arguments @('validate-done-set','--run-dir',$RunDir,'--authorization',$authorization,'--done-root',$doneRoot) -Label 'revalidate audit50 DONE set during pre-selector recovery')
    [void](Invoke-M43A8Audit50Python -ModuleFile $module -Arguments @('validate-claims','--run-dir',$RunDir,'--authorization',$authorization,'--local-claim',$localClaim,'--remote-claim',$remoteClaim) -Label 'revalidate audit50 claims during pre-selector recovery')
    [void](Invoke-M43A8Audit50Python -ModuleFile $module -Arguments @('validate-merged','--run-dir',$RunDir,'--authorization',$authorization,'--local-claim',$localClaim,'--remote-claim',$remoteClaim,'--merged',$merged,'--receipt',$mergeReceipt) -Label 'revalidate audit50 merge during pre-selector recovery')
}

# Phase 3: establish local+remote one-shot selector claim before opening gates.
$localClaim = Join-Path $base 'claims/CONSUMED.json'; $remoteClaim = Join-Path $base 'claims/REMOTE_CONSUMED.json'; $merged = Join-Path $base 'merged/teacher.jsonl'; $mergeReceipt = Join-Path $base 'merged/merge_receipt.json'
$selectorRoot = Join-Path $base 'selector'; New-Item -ItemType Directory -Path $selectorRoot -Force | Out-Null
$selectorClaim = Join-Path $selectorRoot 'CLAIM.json'; $remoteSelectorClaim = Join-Path $selectorRoot 'REMOTE_CLAIM.json'
if (-not (Test-Path -LiteralPath $selectorClaim -PathType Leaf)) {
    [void](Invoke-M43A8Audit50Python -ModuleFile $module -Arguments @('selector-claim','--run-dir',$RunDir,'--authorization',$authorization,'--local-consumption-claim',$localClaim,'--remote-consumption-claim',$remoteClaim,'--merged',$merged,'--merge-receipt',$mergeReceipt,'--output',$selectorClaim) -Label 'claim sole audit50 selector execution')
}
$remoteSelectorUri = "$prefix/claims/audit50_selector.json"
[void](Publish-M43A8Audit50ImmutableObject -Source $selectorClaim -Uri $remoteSelectorUri -ProjectId $ProjectId)
Sync-M43A8Audit50ImmutableRemoteCopy -Uri $remoteSelectorUri -Destination $remoteSelectorClaim
$raw = Invoke-M43A8Audit50Python -ModuleFile $module -Arguments @('select-once','--run-dir',$RunDir,'--development-run-dir',$DevelopmentRunDir,'--selector-claim',$selectorClaim,'--remote-selector-claim',$remoteSelectorClaim) -Label 'evaluate frozen audit50 gates exactly once' -TimeoutSeconds 3600
$selection = $raw.Trim() | ConvertFrom-Json
[void](Invoke-M43A8Audit50Python -ModuleFile $module -Arguments @('validate-selector','--run-dir',$base) -Label 'validate completed audit50 selector')
[pscustomobject]@{
    schema = 'hu_m43_attempt08_audit50_receive_result_v1'
    status = 'exact_50_received_and_selector_executed_once'
    run_name = $RunName
    output_root = $base
    merged = $merged
    decision = (Join-Path $base 'selector/decision.json')
    decision_receipt = (Join-Path $base 'selector/decision_receipt.json')
    selector = $selection
    gate_evaluation_count = 1
    audit_rows_used_for_fit = $false
    current_profile_mutated = $false
    runtime_policy_activated = $false
} | ConvertTo-Json -Depth 12
