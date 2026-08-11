param(
    [string]$ProjectId = 'ofc-solver-485418',
    [string]$Bucket = 'pokerhu-ofc-solver-485418-training',
    [string]$RunName = ('regular-hu-m43-attempt08-development200-' + (Get-Date -Format 'yyyyMMdd-HHmmss')),
    [string]$RunDir,
    [string]$PreflightOutputDir,
    [string]$PreflightSpotRunDir,
    [string]$PreflightSpotJobsRoot,
    [string]$PreflightSpotLaunchAuthorization,
    [string]$PreflightSpotLocalEvidence,
    [string[]]$StartShards = @('0-24'),
    [string]$Zone = 'asia-northeast1-b',
    [string]$MachineType = 'c4-highmem-4',
    [string]$ImageProject = 'debian-cloud',
    [string]$ImageName = 'debian-12-bookworm-v20260609',
    [string]$BootDiskType = 'hyperdisk-balanced',
    [ValidateRange(20, 200)][int]$BootDiskGb = 50,
    [switch]$PackageOnly,
    [switch]$AuthorizeLaunch,
    [switch]$CreateInstances,
    [switch]$ResumePackage,
    [switch]$NoSelfDelete
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$ExpectedImageId = '1449487925682397051'
$ExpectedImageSelfLink = 'https://www.googleapis.com/compute/v1/projects/debian-cloud/global/images/debian-12-bookworm-v20260609'
$ExpectedMachineType = 'c4-highmem-4'
$phaseCount = @($PackageOnly, $AuthorizeLaunch, $CreateInstances).Where({ [bool]$_ }).Count
if ($phaseCount -ne 1) {
    throw 'Choose exactly one Attempt08 phase: PackageOnly, AuthorizeLaunch, or CreateInstances'
}
if ($RunName -notmatch '^[a-z0-9][a-z0-9-]{2,126}[a-z0-9]$') {
    throw 'Attempt08 RunName is not a safe GCP identity'
}
if ($MachineType -ne $ExpectedMachineType) { throw 'Attempt08 development is frozen to c4-highmem-4' }
if ($ImageProject -ne 'debian-cloud' -or $ImageName -ne 'debian-12-bookworm-v20260609') {
    throw 'Attempt08 development requires the immutable authorized Debian image'
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $RunDir) { $RunDir = Join-Path $repoRoot "outputs/gcp_runs/$RunName" }
$RunDir = [IO.Path]::GetFullPath($RunDir)
$expectedRunDir = [IO.Path]::GetFullPath((Join-Path $repoRoot "outputs/gcp_runs/$RunName"))
if (-not [string]::Equals($RunDir, $expectedRunDir, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Attempt08 RunDir must be outputs/gcp_runs/<RunName>'
}
$commonPath = Join-Path $PSScriptRoot 'HuM43Attempt08Spot.Common.ps1'
if (-not $PackageOnly) {
    $frozenScript = Join-Path $RunDir 'package_src/scripts/Start-GcpHuM43Attempt08DevelopmentRun.ps1'
    $commonPath = Join-Path $RunDir 'package_src/scripts/HuM43Attempt08Spot.Common.ps1'
    foreach ($path in @($frozenScript,$commonPath,(Join-Path $RunDir 'package_src/scripts/HuM43Attempt04Spot.Common.ps1'))) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Attempt08 frozen lifecycle source is missing: $path" }
    }
    if ((Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash -ne
        (Get-FileHash -LiteralPath $frozenScript -Algorithm SHA256).Hash) {
        throw 'Attempt08 Start script differs from its frozen packaged copy'
    }
}
. $commonPath

function Resolve-A8Path {
    param([string]$Path, [string]$Label, [switch]$RequireFile)
    return Resolve-M43A4Path -Path $Path -Root $repoRoot -Label $Label -RequireFile:$RequireFile
}
function Invoke-A8Python {
    param([string[]]$Arguments, [string]$Label, [int]$TimeoutSeconds = 1800)
    $moduleRoot = if ($PackageOnly) { $repoRoot } else { Join-Path $RunDir 'package_src' }
    return Invoke-M43A8Python -RepoRoot $moduleRoot -Arguments $Arguments -Label $Label -TimeoutSeconds $TimeoutSeconds
}

if ($PackageOnly) {
if (-not $PreflightOutputDir -or -not $PreflightSpotRunDir -or -not $PreflightSpotJobsRoot) {
    throw 'PackageOnly requires PreflightOutputDir, PreflightSpotRunDir, and PreflightSpotJobsRoot'
}
$preflightOutput = Resolve-A8Path $PreflightOutputDir 'Attempt08 preflight output directory'
$preflightSpotRun = Resolve-A8Path $PreflightSpotRunDir 'Attempt08 preflight Spot run directory'
$preflightJobs = Resolve-A8Path $PreflightSpotJobsRoot 'Attempt08 preflight jobs root'
if (-not $PreflightSpotLaunchAuthorization) {
    $PreflightSpotLaunchAuthorization = Join-Path $preflightSpotRun 'spot_authorization.json'
}
if (-not $PreflightSpotLocalEvidence) {
    $PreflightSpotLocalEvidence = Join-Path $preflightSpotRun 'local_evidence.json'
}
$preflightSpotAuthorization = Resolve-A8Path $PreflightSpotLaunchAuthorization 'Attempt08 preflight Spot authorization' -RequireFile
$preflightSpotEvidence = Resolve-A8Path $PreflightSpotLocalEvidence 'Attempt08 preflight local evidence' -RequireFile
$plan = Resolve-A8Path 'configs/hu_joint_policy_m43_attempt08.json' 'Attempt08 plan' -RequireFile
$preflightPlan = Resolve-A8Path 'configs/hu_joint_policy_m43_attempt08_preflight.json' 'Attempt08 preflight plan' -RequireFile
$aggregate = Resolve-A8Path (Join-Path $preflightOutput 'aggregate.json') 'Attempt08 preflight aggregate' -RequireFile
$executionEvidence = Resolve-A8Path (Join-Path $preflightOutput 'preflight_execution_evidence.json') 'Attempt08 preflight execution evidence' -RequireFile
$finalization = Resolve-A8Path (Join-Path $preflightOutput 'finalization.json') 'Attempt08 preflight finalization' -RequireFile
$developmentOpen = Resolve-A8Path (Join-Path $preflightOutput 'development_open_authorization.json') 'Attempt08 development-open authorization' -RequireFile
$preflightSource = Resolve-A8Path 'outputs/hu_joint_policy/m43_attempt06_search_quality/regular-hu-m43-attempt06-preflight-final-20260714-1154/merged/teacher.jsonl' 'Attempt08 preflight source' -RequireFile
$model = Resolve-A8Path 'outputs/hu_joint_policy/m43_attempt05_old_dev900_architecture_once/lambda_rank_candidate.pkl' 'Attempt08 Lambda model' -RequireFile
$startup = Resolve-A8Path 'scripts/startup_hu_m43_attempt08_development.sh' 'Attempt08 development startup' -RequireFile

$proofArguments = [Collections.Generic.List[string]]::new()
$slotNames = @('root0_batch_a','root0_batch_b','root0_scalar','root1_batch','root2_batch')
for ($index = 0; $index -lt $slotNames.Count; $index++) {
    $slot = $slotNames[$index]
    $proof = Resolve-A8Path (Join-Path $preflightJobs ("job_{0:d3}_{1}/proof.json" -f $index, $slot)) "Attempt08 proof $slot" -RequireFile
    $proofArguments.Add('--' + ($slot -replace '_','-'))
    $proofArguments.Add($proof)
}

    $arguments = [Collections.Generic.List[string]]::new()
    @(
        '-B','-m','ofc_regular.hu_m43_attempt08_spot','package',
        '--repo-root',$repoRoot,'--run-dir',$RunDir,'--run-name',$RunName,
        '--plan',$plan,'--preflight-plan',$preflightPlan,
        '--preflight-aggregate',$aggregate,
        '--preflight-execution-evidence',$executionEvidence,
        '--preflight-spot-run-dir',$preflightSpotRun,
        '--preflight-spot-launch-authorization',$preflightSpotAuthorization,
        '--preflight-spot-local-evidence',$preflightSpotEvidence,
        '--preflight-spot-jobs-root',$preflightJobs,
        '--preflight-finalization',$finalization,
        '--preflight-source',$preflightSource,
        '--development-open-authorization',$developmentOpen,
        '--model',$model,'--startup',$startup
    ) | ForEach-Object { $arguments.Add([string]$_) }
    foreach ($item in $proofArguments) { $arguments.Add($item) }
    if ($ResumePackage) { $arguments.Add('--resume-existing') }
    $raw = Invoke-A8Python -Arguments @($arguments) -Label 'package Attempt08 development without GCP or root execution' -TimeoutSeconds 3600
    $result = $raw.Trim() | ConvertFrom-Json
    if ($result.status -ne 'packaged_without_root_or_gcloud' -or
        [int]$result.total_shards -ne 200 -or $result.fresh_root_opened -ne $false -or
        $result.teacher_executed -ne $false -or $result.gcloud_invoked -ne $false -or
        $result.instances_created -ne $false -or $result.current_profile_mutated -ne $false) {
        throw 'Attempt08 PackageOnly boundary changed'
    }
    $result | ConvertTo-Json -Depth 8
    return
}

$manifestPath = Join-Path $RunDir 'manifest.json'
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw 'Attempt08 frozen package is missing; run PackageOnly first'
}
$launchPath = Join-Path $RunDir 'launch_authorization.json'
if ($AuthorizeLaunch) {
    if (Test-Path -LiteralPath $launchPath) {
        throw 'Attempt08 launch authorization already exists and is immutable'
    }
    $raw = Invoke-A8Python -Arguments @(
        '-B','-m','ofc_regular.hu_m43_attempt08_spot','authorize-launch',
        '--run-dir',$RunDir,'--output',$launchPath
    ) -Label 'authorize frozen Attempt08 development package'
    $result = $raw.Trim() | ConvertFrom-Json
    if ($result.spot_authorized -ne $true -or $result.development_started -ne $false -or
        $result.current_profile_mutated -ne $false -or $result.runtime_policy_activated -ne $false) {
        throw 'Attempt08 launch authorization boundary changed'
    }
    $result | ConvertTo-Json -Depth 8
    return
}

if (-not (Test-Path -LiteralPath $launchPath -PathType Leaf)) {
    throw 'Attempt08 immutable launch authorization is missing; run AuthorizeLaunch first'
}
[void](Invoke-A8Python -Arguments @(
    '-B','-m','ofc_regular.hu_m43_attempt08_spot','validate-launch',
    '--run-dir',$RunDir,'--authorization',$launchPath
) -Label 'validate Attempt08 package and launch authorization' -TimeoutSeconds 3600)

$imageRaw = Invoke-M43A4Gcloud -Arguments @(
    'compute','images','describe',$ImageName,'--project',$ImageProject,'--format=json'
) -TimeoutSeconds 120 -Label 'describe immutable Attempt08 boot image'
$image = $imageRaw | ConvertFrom-Json
if ([string]$image.name -ne $ImageName -or [string]$image.id -ne $ExpectedImageId -or
    [string]$image.selfLink -ne $ExpectedImageSelfLink) {
    throw 'Attempt08 immutable boot image identity changed'
}

$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$schedulePath = Join-Path $RunDir 'shards_manifest.jsonl'
$sourcePath = Join-Path $RunDir 'ofc_regular_hu_m43_attempt08_development_source.zip'
$frozenStartup = Join-Path $RunDir 'startup_hu_m43_attempt08_development.sh'
if ((Get-M43A4Sha256 $frozenStartup) -ne [string]$manifest.startup_sha256) {
    throw 'Attempt08 frozen startup hash disagrees with package manifest'
}
$prefix = "gs://$Bucket/runs/$RunName"
foreach ($publish in @(
    [pscustomobject]@{ source = $manifestPath; uri = "$prefix/manifest.json" },
    [pscustomobject]@{ source = $launchPath; uri = "$prefix/source/launch_authorization.json" },
    [pscustomobject]@{ source = $schedulePath; uri = "$prefix/source/shards_manifest.jsonl" },
    [pscustomobject]@{ source = $sourcePath; uri = "$prefix/source/ofc_regular_hu_m43_attempt08_development_source.zip" }
)) {
    [void](Publish-M43A8ImmutableObject -Source $publish.source -Uri $publish.uri -ProjectId $ProjectId)
}

$selection = @(Expand-M43A8ShardSelection -Values $StartShards -MaxCount 25)
$schedule = @(Get-Content -LiteralPath $schedulePath | Where-Object { $_ } | ForEach-Object { $_ | ConvertFrom-Json })
if ($schedule.Count -ne 200) { throw 'Attempt08 schedule must contain exactly 200 shards' }
$vmPrefix = ConvertTo-M43A8VmPrefix $RunName
$created = [Collections.Generic.List[object]]::new()
foreach ($shard in $selection) {
    $spec = $schedule[$shard]
    if ([int]$spec.shard -ne $shard -or [int]$spec.root_index -ne $shard) {
        throw "Attempt08 schedule mapping changed at shard $shard"
    }
    $doneUri = "$prefix/results/$($spec.output_prefix)/DONE.json"
    if (Test-M43A4GcsObject -Uri $doneUri -ProjectId $ProjectId) {
        throw "Attempt08 DONE already exists for shard $shard; use status/receive validation, never blind launch skip"
    }
    $instance = '{0}-s{1:d3}' -f $vmPrefix, $shard
    $existingRaw = Invoke-M43A4GcloudProcess -Arguments @(
        'compute','instances','describe',$instance,'--zone',$Zone,'--project',$ProjectId,'--format=json'
    ) -TimeoutSeconds 60 -Label "describe Attempt08 instance $instance"
    if (-not $existingRaw.timed_out -and $existingRaw.exit_code -eq 0) {
        throw "Attempt08 instance already exists: $instance; blind instance reuse is forbidden"
    }
    $describeFailure = ([string]$existingRaw.stdout) + ([string]$existingRaw.stderr)
    if ($existingRaw.timed_out -or $describeFailure -notmatch '(?i)not found|was not found|404') {
        throw "Unable to prove Attempt08 instance absence: $instance"
    }
    $selfDelete = if ($NoSelfDelete) { '0' } else { '1' }
    $metadata = @(
        "RUN_NAME=$RunName","PROJECT_ID=$ProjectId","BUCKET=$Bucket","SHARD=$shard",
        "SOURCE_URI=$prefix/source/ofc_regular_hu_m43_attempt08_development_source.zip",
        "SOURCE_SHA256=$($manifest.source_zip_sha256)",
        "MANIFEST_SHA256=$(Get-M43A4Sha256 $manifestPath)",
        "LAUNCH_AUTHORIZATION_SHA256=$(Get-M43A4Sha256 $launchPath)",
        "SCHEDULE_SHA256=$($manifest.schedule_sha256)",
        "STARTUP_SHA256=$($manifest.startup_sha256)",
        "SELF_DELETE=$selfDelete"
    ) -join ','
    [void](Invoke-M43A4Gcloud -Arguments @(
        'compute','instances','create',$instance,'--project',$ProjectId,'--zone',$Zone,
        '--machine-type',$MachineType,'--provisioning-model','SPOT',
        '--instance-termination-action','DELETE','--maintenance-policy','TERMINATE',
        '--no-restart-on-failure','--scopes','cloud-platform','--image',$ImageName,
        '--image-project',$ImageProject,'--boot-disk-type',$BootDiskType,
        '--boot-disk-size',("${BootDiskGb}GB"),'--metadata',$metadata,
        '--metadata-from-file',("startup-script=$frozenStartup"),'--format=json'
    ) -TimeoutSeconds 600 -Label "create Attempt08 shard $shard")
    $instanceJson = (Invoke-M43A4Gcloud -Arguments @(
        'compute','instances','describe',$instance,'--zone',$Zone,'--project',$ProjectId,'--format=json'
    ) -TimeoutSeconds 120 -Label "verify Attempt08 instance $instance") | ConvertFrom-Json
    $diskName = ([string]$instanceJson.disks[0].source).Split('/')[-1]
    $disk = (Invoke-M43A4Gcloud -Arguments @(
        'compute','disks','describe',$diskName,'--zone',$Zone,'--project',$ProjectId,'--format=json'
    ) -TimeoutSeconds 120 -Label "verify Attempt08 boot disk $diskName") | ConvertFrom-Json
    if ([string]$disk.sourceImageId -ne $ExpectedImageId -or
        -not ([string]$disk.sourceImage).EndsWith("/projects/$ImageProject/global/images/$ImageName")) {
        throw "Attempt08 instance $instance boot image is not the authorized immutable image"
    }
    $created.Add([pscustomobject]@{ shard=$shard; instance=$instance; output_prefix=$spec.output_prefix })
}

[pscustomobject]@{
    schema = 'hu_m43_attempt08_development_launch_result_v1'
    status = 'selected_wave_created'
    run_name = $RunName
    selected_shards = $selection
    created = @($created)
    machine_type = $MachineType
    image_name = $ImageName
    image_id = $ExpectedImageId
    max_wave_shards = 25
    current_profile_mutated = $false
    runtime_policy_activated = $false
} | ConvertTo-Json -Depth 8
