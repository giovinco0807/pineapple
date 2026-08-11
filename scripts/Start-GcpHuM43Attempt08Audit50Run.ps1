param(
    [string]$ProjectId = 'ofc-solver-485418',
    [string]$Bucket = 'pokerhu-ofc-solver-485418-training',
    [string]$RunName = ('regular-hu-m43-attempt08-audit50-' + (Get-Date -Format 'yyyyMMdd-HHmmss')),
    [string]$DevelopmentRunName = 'regular-hu-m43-attempt08-development200-finalprop-20260714-215952',
    [string]$RunDir,
    [string]$DevelopmentRunDir,
    [string]$DevelopmentOutputDir,
    [string]$DevelopmentPassFreeze,
    [string]$DevelopmentDecision,
    [string]$DevelopmentSelectorReceipt,
    [string[]]$StartShards = @('0-24'),
    [string]$Zone = 'asia-northeast1-b',
    [string]$MachineType = 'c4-highmem-4',
    [string]$ImageProject = 'debian-cloud',
    [string]$ImageName = 'debian-12-bookworm-v20260609',
    [string]$BootDiskType = 'hyperdisk-balanced',
    [ValidateRange(20, 200)][int]$BootDiskGb = 50,
    [switch]$PackageOnly,
    [switch]$AuthorizeAudit,
    [switch]$AuthorizeLaunch,
    [switch]$CreateInstances,
    [switch]$ResumePackage,
    [switch]$NoSelfDelete
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$ExpectedImageId = '1449487925682397051'
$ExpectedImageSelfLink = 'https://www.googleapis.com/compute/v1/projects/debian-cloud/global/images/debian-12-bookworm-v20260609'
$phaseCount = @($PackageOnly,$AuthorizeAudit,$AuthorizeLaunch,$CreateInstances).Where({ [bool]$_ }).Count
if ($phaseCount -ne 1) { throw 'Choose exactly one audit50 phase: PackageOnly, AuthorizeAudit, AuthorizeLaunch, or CreateInstances' }
if ($RunName -notmatch '^[a-z0-9][a-z0-9-]{2,126}[a-z0-9]$') { throw 'Attempt08 audit50 RunName is unsafe' }
if ($DevelopmentRunName -notmatch '^[a-z0-9][a-z0-9-]{2,126}[a-z0-9]$') { throw 'Attempt08 development RunName is unsafe' }
if ($MachineType -ne 'c4-highmem-4') { throw 'Attempt08 audit50 is frozen to c4-highmem-4' }
if ($ImageProject -ne 'debian-cloud' -or $ImageName -ne 'debian-12-bookworm-v20260609') { throw 'Attempt08 audit50 requires the immutable authorized image' }

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $RunDir) { $RunDir = Join-Path $repoRoot "outputs/gcp_runs/$RunName" }
if (-not $DevelopmentRunDir) { $DevelopmentRunDir = Join-Path $repoRoot "outputs/gcp_runs/$DevelopmentRunName" }
$RunDir = [IO.Path]::GetFullPath($RunDir)
$DevelopmentRunDir = (Resolve-Path $DevelopmentRunDir).Path
$expectedRunDir = [IO.Path]::GetFullPath((Join-Path $repoRoot "outputs/gcp_runs/$RunName"))
if (-not [string]::Equals($RunDir,$expectedRunDir,[StringComparison]::OrdinalIgnoreCase)) { throw 'Attempt08 audit50 RunDir must be outputs/gcp_runs/<RunName>' }
$script:Audit50DevelopmentRunDir = $DevelopmentRunDir
$commonPath = if ($PackageOnly) { Join-Path $PSScriptRoot 'HuM43Attempt08Audit50Spot.Common.ps1' } else { Join-Path $RunDir 'overlay_src/scripts/HuM43Attempt08Audit50Spot.Common.ps1' }
. $commonPath

$workspaceModule = Join-Path $repoRoot 'src/ofc_regular/hu_m43_attempt08_audit50_spot.py'
$module = if ($PackageOnly) { $workspaceModule } else { Join-Path $RunDir 'overlay_src/src/ofc_regular/hu_m43_attempt08_audit50_spot.py' }
if (-not $PackageOnly) {
    $frozenScript = Join-Path $RunDir 'overlay_src/scripts/Start-GcpHuM43Attempt08Audit50Run.ps1'
    $frozenCommon = Join-Path $RunDir 'overlay_src/scripts/HuM43Attempt08Audit50Spot.Common.ps1'
    foreach ($path in @($module,$frozenScript,$frozenCommon)) { if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Attempt08 audit50 frozen lifecycle source missing: $path" } }
    if ((Get-FileHash $PSCommandPath -Algorithm SHA256).Hash -ne (Get-FileHash $frozenScript -Algorithm SHA256).Hash) { throw 'Attempt08 audit50 Start script differs from frozen overlay' }
}

if ($PackageOnly) {
    $auditPlan = Join-Path $repoRoot 'configs/hu_joint_policy_m43_attempt08_audit50.json'
    $arguments = @('package','--repo-root',$repoRoot,'--run-dir',$RunDir,'--run-name',$RunName,'--development-run-dir',$DevelopmentRunDir,'--audit-plan',$auditPlan)
    if ($ResumePackage) { $arguments += '--resume-existing' }
    $raw = Invoke-M43A8Audit50Python -ModuleFile $module -Arguments $arguments -Label 'package Attempt08 audit50 overlay without authorization/root/GCP' -TimeoutSeconds 3600
    $result = $raw.Trim() | ConvertFrom-Json
    if ($result.status -ne 'packaged_without_audit_authorization_root_or_gcloud' -or [int]$result.total_shards -ne 50 -or $result.audit_authorized -ne $false -or $result.fresh_root_opened -ne $false -or $result.teacher_executed -ne $false -or $result.gcloud_invoked -ne $false -or $result.current_profile_mutated -ne $false) { throw 'Attempt08 audit50 PackageOnly boundary changed' }
    $result | ConvertTo-Json -Depth 8
    return
}

$manifestPath = Join-Path $RunDir 'manifest.json'
if (-not (Test-Path $manifestPath -PathType Leaf)) { throw 'Attempt08 audit50 package is missing; run PackageOnly first' }
if (-not $DevelopmentOutputDir) { $DevelopmentOutputDir = Join-Path $repoRoot "outputs/hu_joint_policy/m43_attempt08_development/$DevelopmentRunName" }
if (-not $DevelopmentPassFreeze) { $DevelopmentPassFreeze = Join-Path $DevelopmentOutputDir 'selector/development_pass_freeze.json' }
if (-not $DevelopmentDecision) { $DevelopmentDecision = Join-Path $DevelopmentOutputDir 'selector/decision.json' }
if (-not $DevelopmentSelectorReceipt) { $DevelopmentSelectorReceipt = Join-Path $DevelopmentOutputDir 'selector/decision_receipt.json' }

if ($AuthorizeAudit) {
    foreach ($path in @($DevelopmentPassFreeze,$DevelopmentDecision,$DevelopmentSelectorReceipt)) { if (-not (Test-Path $path -PathType Leaf)) { throw "Attempt08 development GO artifact missing: $path" } }
    $core = Join-Path $RunDir 'core_audit_authorization.json'
    $outer = Join-Path $RunDir 'audit_open_authorization.json'
    $raw = Invoke-M43A8Audit50Python -ModuleFile $module -Arguments @('authorize-audit','--run-dir',$RunDir,'--development-run-dir',$DevelopmentRunDir,'--development-pass-freeze',$DevelopmentPassFreeze,'--development-decision',$DevelopmentDecision,'--selector-receipt',$DevelopmentSelectorReceipt,'--core-output',$core,'--outer-output',$outer) -Label 'authorize audit50 after immutable development GO' -TimeoutSeconds 3600
    $result = $raw.Trim() | ConvertFrom-Json
    if ($result.audit_authorized -ne $true -or $result.audit_started -ne $false -or $result.fit_performed -ne $false -or $result.current_profile_mutated -ne $false) { throw 'Attempt08 audit50 authorization boundary changed' }
    $result | ConvertTo-Json -Depth 8
    return
}

$outerPath = Join-Path $RunDir 'audit_open_authorization.json'
if (-not (Test-Path $outerPath -PathType Leaf)) { throw 'Attempt08 audit50 open authorization is missing; run AuthorizeAudit first' }
$launchPath = Join-Path $RunDir 'launch_authorization.json'
if ($AuthorizeLaunch) {
    if (Test-Path $launchPath) { throw 'Attempt08 audit50 launch authorization already exists and is immutable' }
    $raw = Invoke-M43A8Audit50Python -ModuleFile $module -Arguments @('authorize-launch','--run-dir',$RunDir,'--development-run-dir',$DevelopmentRunDir,'--output',$launchPath) -Label 'authorize frozen Attempt08 audit50 Spot launch' -TimeoutSeconds 3600
    $result = $raw.Trim() | ConvertFrom-Json
    if ($result.spot_authorized -ne $true -or $result.audit_started -ne $false -or $result.current_profile_mutated -ne $false) { throw 'Attempt08 audit50 launch authorization boundary changed' }
    $result | ConvertTo-Json -Depth 8
    return
}

if (-not (Test-Path $launchPath -PathType Leaf)) { throw 'Attempt08 audit50 launch authorization missing; run AuthorizeLaunch first' }
[void](Invoke-M43A8Audit50Python -ModuleFile $module -Arguments @('validate-launch','--run-dir',$RunDir,'--development-run-dir',$DevelopmentRunDir,'--authorization',$launchPath) -Label 'validate Attempt08 audit50 launch chain' -TimeoutSeconds 3600)
$image = (Invoke-M43A4Gcloud -Arguments @('compute','images','describe',$ImageName,'--project',$ImageProject,'--format=json') -TimeoutSeconds 120 -Label 'describe immutable audit50 image') | ConvertFrom-Json
if ([string]$image.name -ne $ImageName -or [string]$image.id -ne $ExpectedImageId -or [string]$image.selfLink -ne $ExpectedImageSelfLink) { throw 'Attempt08 audit50 boot image identity changed' }

$manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json
$schedulePath = Join-Path $RunDir 'shards_manifest.jsonl'
$sourcePath = Join-Path $RunDir 'ofc_regular_hu_m43_attempt08_audit50_overlay.zip'
$startup = Join-Path $RunDir 'startup_hu_m43_attempt08_audit50.sh'
$prefix = "gs://$Bucket/runs/$RunName"
$publish = @(
    @{source=$manifestPath;uri="$prefix/manifest.json"},
    @{source=$launchPath;uri="$prefix/source/launch_authorization.json"},
    @{source=$outerPath;uri="$prefix/source/audit_open_authorization.json"},
    @{source=(Join-Path $RunDir 'core_audit_authorization.json');uri="$prefix/source/core_audit_authorization.json"},
    @{source=(Join-Path $RunDir 'development_pass_freeze.json');uri="$prefix/source/development_pass_freeze.json"},
    @{source=(Join-Path $RunDir 'development_decision.json');uri="$prefix/source/development_decision.json"},
    @{source=(Join-Path $RunDir 'development_selector_receipt.json');uri="$prefix/source/development_selector_receipt.json"},
    @{source=$schedulePath;uri="$prefix/source/shards_manifest.jsonl"},
    @{source=(Join-Path $RunDir 'source_closure_manifest.json');uri="$prefix/source/source_closure_manifest.json"},
    @{source=$sourcePath;uri="$prefix/source/ofc_regular_hu_m43_attempt08_audit50_overlay.zip"}
)
foreach ($item in $publish) { [void](Publish-M43A8Audit50ImmutableObject -Source $item.source -Uri $item.uri -ProjectId $ProjectId) }

$devPrefix = "gs://$Bucket/runs/$DevelopmentRunName"
foreach ($uri in @("$devPrefix/manifest.json","$devPrefix/source/launch_authorization.json","$devPrefix/source/shards_manifest.jsonl","$devPrefix/source/ofc_regular_hu_m43_attempt08_development_source.zip")) { if (-not (Test-M43A4GcsObject -Uri $uri -ProjectId $ProjectId)) { throw "Frozen development remote object is missing: $uri" } }

$selection = @(Expand-M43A8Audit50ShardSelection -Values $StartShards)
$schedule = @(Get-Content $schedulePath | Where-Object { $_ } | ForEach-Object { $_ | ConvertFrom-Json })
if ($schedule.Count -ne 50) { throw 'Attempt08 audit50 schedule must contain exactly 50 shards' }
$vmPrefix = ConvertTo-M43A8VmPrefix $RunName
$created = [Collections.Generic.List[object]]::new()
$coreAuthSha = (Get-FileHash (Join-Path $RunDir 'core_audit_authorization.json') -Algorithm SHA256).Hash.ToLower()
foreach ($shard in $selection) {
    $spec = $schedule[$shard]
    if ([int]$spec.shard -ne $shard -or [int]$spec.root_index -ne (200 + $shard)) { throw "Attempt08 audit50 schedule mapping changed at shard $shard" }
    $doneUri = "$prefix/results/$($spec.output_prefix)/DONE.json"
    if (Test-M43A4GcsObject -Uri $doneUri -ProjectId $ProjectId) { throw "Attempt08 audit50 DONE already exists for shard $shard; validate it instead of blind launch skip" }
    $instance = '{0}-s{1:d3}' -f $vmPrefix,$shard
    $existing = Invoke-M43A4GcloudProcess -Arguments @('compute','instances','describe',$instance,'--zone',$Zone,'--project',$ProjectId,'--format=json') -TimeoutSeconds 60 -Label "describe audit50 instance $instance"
    if (-not $existing.timed_out -and $existing.exit_code -eq 0) { throw "Attempt08 audit50 instance already exists: $instance" }
    $failure = ([string]$existing.stdout) + ([string]$existing.stderr)
    if ($existing.timed_out -or $failure -notmatch '(?i)not found|was not found|404') { throw "Unable to prove audit50 instance absence: $instance" }
    $selfDelete = if ($NoSelfDelete) { '0' } else { '1' }
    $metadata = @(
        "RUN_NAME=$RunName","DEV_RUN_NAME=$DevelopmentRunName","PROJECT_ID=$ProjectId","BUCKET=$Bucket","SHARD=$shard",
        "AUDIT_SOURCE_SHA256=$($manifest.overlay_source_zip_sha256)","AUDIT_MANIFEST_SHA256=$((Get-FileHash $manifestPath -Algorithm SHA256).Hash.ToLower())",
        "AUDIT_LAUNCH_SHA256=$((Get-FileHash $launchPath -Algorithm SHA256).Hash.ToLower())","AUDIT_OPEN_SHA256=$((Get-FileHash $outerPath -Algorithm SHA256).Hash.ToLower())",
        "CORE_AUTH_SHA256=$coreAuthSha",
        "FREEZE_SHA256=$((Get-FileHash (Join-Path $RunDir 'development_pass_freeze.json') -Algorithm SHA256).Hash.ToLower())",
        "DECISION_SHA256=$((Get-FileHash (Join-Path $RunDir 'development_decision.json') -Algorithm SHA256).Hash.ToLower())","RECEIPT_SHA256=$((Get-FileHash (Join-Path $RunDir 'development_selector_receipt.json') -Algorithm SHA256).Hash.ToLower())",
        "AUDIT_SCHEDULE_SHA256=$($manifest.schedule_sha256)","STARTUP_SHA256=$($manifest.startup_sha256)",
        "DEV_MANIFEST_SHA256=$($manifest.development_manifest_sha256)","DEV_LAUNCH_SHA256=$($manifest.development_launch_authorization_sha256)",
        "DEV_SOURCE_SHA256=$($manifest.development_source_zip_sha256)","DEV_SCHEDULE_SHA256=$($manifest.development_schedule_sha256)","DEV_STARTUP_SHA256=$($manifest.development_startup_sha256)","SELF_DELETE=$selfDelete"
    ) -join ','
    [void](Invoke-M43A4Gcloud -Arguments @('compute','instances','create',$instance,'--project',$ProjectId,'--zone',$Zone,'--machine-type',$MachineType,'--provisioning-model','SPOT','--instance-termination-action','DELETE','--maintenance-policy','TERMINATE','--no-restart-on-failure','--scopes','cloud-platform','--image',$ImageName,'--image-project',$ImageProject,'--boot-disk-type',$BootDiskType,'--boot-disk-size',("${BootDiskGb}GB"),'--metadata',$metadata,'--metadata-from-file',("startup-script=$startup"),'--format=json') -TimeoutSeconds 600 -Label "create audit50 shard $shard")
    $created.Add([pscustomobject]@{shard=$shard;root_index=(200+$shard);instance=$instance;output_prefix=$spec.output_prefix})
}
[pscustomobject]@{schema='hu_m43_attempt08_audit50_launch_result_v1';status='selected_wave_created';run_name=$RunName;selected_shards=$selection;created=@($created);machine_type=$MachineType;image_name=$ImageName;image_id=$ExpectedImageId;max_wave_shards=25;current_profile_mutated=$false;runtime_policy_activated=$false} | ConvertTo-Json -Depth 8
