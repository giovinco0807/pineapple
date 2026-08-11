param(
    [string]$ProjectId = 'ofc-solver-485418',
    [string]$Bucket = 'pokerhu-ofc-solver-485418-training',
    [string]$RunName = ('regular-hu-m43-attempt08-preflight-' + (Get-Date -Format 'yyyyMMdd-HHmmss')),
    [string]$RunDir,
    [string]$Zone = 'asia-northeast1-b',
    [string[]]$Jobs = @('0-4'),
    [switch]$PackageOnly,
    [switch]$CreateAuthorization,
    [switch]$CreateInstances,
    [switch]$ResumePackage,
    [switch]$NoSelfDelete
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $RunDir) { $RunDir = Join-Path $RepoRoot "outputs/gcp_runs/$RunName" }
$RunDir = [IO.Path]::GetFullPath($RunDir)
$ExpectedRunDir=[IO.Path]::GetFullPath((Join-Path $RepoRoot "outputs/gcp_runs/$RunName"))
if($RunDir -cne $ExpectedRunDir){throw 'RunDir must be the exact repository run directory'}
$modes = @(
    @($PackageOnly.IsPresent, $CreateAuthorization.IsPresent, $CreateInstances.IsPresent) |
        Where-Object { $_ }
)
if ($modes.Count -ne 1) { throw 'Choose exactly one of PackageOnly, CreateAuthorization, or CreateInstances' }
if ($RunName -notmatch '^[a-z0-9][a-z0-9-]{2,126}[a-z0-9]$') { throw 'Unsafe RunName' }

function Invoke-Python {
    param(
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$SourceRoot
    )
    if (-not (Test-Path -LiteralPath $SourceRoot -PathType Container)) {
        throw "Python source root is missing: $SourceRoot"
    }
    $oldPythonPath = $env:PYTHONPATH
    $oldNoBytecode = $env:PYTHONDONTWRITEBYTECODE
    $env:PYTHONPATH = $SourceRoot
    $env:PYTHONDONTWRITEBYTECODE = '1'
    try {
        & python @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "python failed: $($Arguments -join ' ')"
        }
    }
    finally {
        $env:PYTHONPATH = $oldPythonPath
        $env:PYTHONDONTWRITEBYTECODE = $oldNoBytecode
    }
}
function Hash([string]$Path) { Get-M43A4Sha256 -Path $Path }
function Publish-Once([string]$Path, [string]$Uri) {
    [void](Publish-M43A8ImmutableObject -Source $Path -Uri $Uri -ProjectId $ProjectId)
}
function Expand-Jobs([string[]]$Values) {
    $set = [Collections.Generic.SortedSet[int]]::new()
    foreach ($raw in $Values) { foreach ($token in ($raw -split ',')) {
        if ($token -match '^(\d+)-(\d+)$') {
            $first=[int]$Matches[1]; $last=[int]$Matches[2]
            if($first -gt $last){throw "Reversed job range: $token"}
            for ($i=$first; $i -le $last; $i++) { [void]$set.Add($i) }
        }
        elseif ($token -match '^\d+$') { [void]$set.Add([int]$token) } else { throw "Bad job: $token" }
    }}
    foreach ($i in $set) { if ($i -lt 0 -or $i -gt 4) { throw "Job outside 0..4: $i" } }
    if($set.Count -eq 0){throw 'At least one job is required'}
    @($set)
}

$Manifest = Join-Path $RunDir 'manifest.json'
$Evidence = Join-Path $RunDir 'local_evidence.json'
$Authorization = Join-Path $RunDir 'spot_authorization.json'
if ($PackageOnly) {
    $args = @('-m','ofc_regular.hu_m43_attempt08_preflight_spot','package','--repo-root',$RepoRoot,
              '--run-dir',$RunDir,'--run-name',$RunName)
    if ($ResumePackage) { $args += '--resume-existing' }
    Invoke-Python -Arguments $args -SourceRoot (Join-Path $RepoRoot 'src')
    Write-Host "Package-only complete. No gcloud call and no teacher execution: $RunDir"
    exit 0
}
if (-not (Test-Path -LiteralPath $Manifest)) { throw 'Package manifest is missing; run PackageOnly first' }
$FrozenSourceRoot = Join-Path $RunDir 'package_src/src'
if (-not (Test-Path -LiteralPath $FrozenSourceRoot -PathType Container)) {
    throw 'Frozen package source is missing; run PackageOnly first'
}
$FrozenScript = Join-Path $RunDir 'package_src/scripts/Start-GcpHuM43Attempt08PreflightRun.ps1'
if (-not (Test-Path -LiteralPath $FrozenScript -PathType Leaf) -or
    (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash -cne
    (Get-FileHash -LiteralPath $FrozenScript -Algorithm SHA256).Hash) {
    throw 'Start-GcpHuM43Attempt08PreflightRun.ps1 differs from its frozen packaged copy'
}
$FrozenCommon08 = Join-Path $RunDir 'package_src/scripts/HuM43Attempt08Spot.Common.ps1'
$FrozenCommon04 = Join-Path $RunDir 'package_src/scripts/HuM43Attempt04Spot.Common.ps1'
foreach ($binding in @(
    @{
        outer = Join-Path $RepoRoot 'scripts/HuM43Attempt08Spot.Common.ps1'
        frozen = $FrozenCommon08
        label = 'HuM43Attempt08Spot.Common.ps1'
    },
    @{
        outer = Join-Path $RepoRoot 'scripts/HuM43Attempt04Spot.Common.ps1'
        frozen = $FrozenCommon04
        label = 'HuM43Attempt04Spot.Common.ps1'
    }
)) {
    if (-not (Test-Path -LiteralPath $binding.frozen -PathType Leaf) -or
        (Get-FileHash -LiteralPath $binding.outer -Algorithm SHA256).Hash -cne
        (Get-FileHash -LiteralPath $binding.frozen -Algorithm SHA256).Hash) {
        throw "$($binding.label) differs from its frozen packaged copy"
    }
}
. $FrozenCommon08
$manifestObject = Get-Content -LiteralPath $Manifest -Raw | ConvertFrom-Json
if ([string]$manifestObject.run_name -cne $RunName) { throw 'Package run_name changed' }
if ($CreateAuthorization) {
    Invoke-Python -Arguments @(
        '-m','ofc_regular.hu_m43_attempt08_preflight_spot','authorize',
        '--manifest',$Manifest,'--evidence-output',$Evidence,'--output',$Authorization
    ) -SourceRoot $FrozenSourceRoot
    Write-Host "Immutable five-job launch authorization created: $Authorization"
    exit 0
}
if (-not (Test-Path -LiteralPath $Authorization) -or -not (Test-Path -LiteralPath $Evidence)) {
    throw 'CreateAuthorization must complete before CreateInstances'
}
Invoke-Python -Arguments @(
    '-m','ofc_regular.hu_m43_attempt08_preflight_spot','validate-launch',
    '--manifest',$Manifest,'--authorization',$Authorization,'--local-evidence',$Evidence
) -SourceRoot $FrozenSourceRoot

$authObject = Get-Content -LiteralPath $Authorization -Raw | ConvertFrom-Json
if ($manifestObject.machine_type -ne 'c4-highmem-4' -or
    [string]$manifestObject.run_name -cne $RunName -or
    $manifestObject.gcp_image_name -ne 'debian-12-bookworm-v20260609' -or
    [string]$manifestObject.gcp_image_id -ne '1449487925682397051' -or
    $authObject.spot_authorized -ne $true -or
    [string]$authObject.manifest_sha256 -ne (Hash $Manifest)) { throw 'Package/authorization identity changed' }

$imageResult = Invoke-M43A4GcloudProcess `
    -Arguments @(
        'compute','images','describe','debian-12-bookworm-v20260609',
        '--project','debian-cloud','--format=json'
    ) `
    -TimeoutSeconds 60 -Label 'describe immutable Attempt08 boot image'
if ($imageResult.timed_out -or $imageResult.exit_code -ne 0) {
    throw 'Unable to describe immutable Attempt08 boot image'
}
$imageObject = ([string]$imageResult.stdout) | ConvertFrom-Json
if ([string]$imageObject.name -ne 'debian-12-bookworm-v20260609' -or
    [string]$imageObject.id -ne '1449487925682397051' -or
    -not ([string]$imageObject.selfLink).EndsWith('/projects/debian-cloud/global/images/debian-12-bookworm-v20260609')) {
    throw 'Immutable Attempt08 boot image identity changed'
}

$Prefix = "gs://$Bucket/runs/$RunName"
Publish-Once (Join-Path $RunDir 'source.zip') "$Prefix/source/source.zip"
Publish-Once $Manifest "$Prefix/manifest.json"
Publish-Once (Join-Path $RunDir 'shards_manifest.jsonl') "$Prefix/source/shards_manifest.jsonl"
Publish-Once (Join-Path $RunDir 'startup_hu_m43_attempt08_preflight.sh') "$Prefix/source/startup_hu_m43_attempt08_preflight.sh"
Publish-Once $Evidence "$Prefix/source/local_evidence.json"
Publish-Once $Authorization "$Prefix/source/spot_authorization.json"

$selected = @(Expand-Jobs $Jobs)
$schedule = @(Get-Content -LiteralPath (Join-Path $RunDir 'shards_manifest.jsonl') | ForEach-Object { $_ | ConvertFrom-Json })
if ($schedule.Count -ne 5) { throw 'Attempt08 preflight schedule must contain exactly five jobs' }
$selfDelete = if ($NoSelfDelete) { '0' } else { '1' }
foreach ($index in $selected) {
    $spec = $schedule[$index]
    if ([int]$spec.job_index -ne $index -or [string]$spec.output_prefix -notmatch '^job_00[0-4]_(root0_batch_a|root0_batch_b|root0_scalar|root1_batch|root2_batch)$') {
        throw "Attempt08 preflight schedule mapping changed at job $index"
    }
    $doneUri = "$Prefix/results/$($spec.output_prefix)/DONE.json"
    if (Test-M43A4GcsObject -Uri $doneUri -ProjectId $ProjectId) {
        throw "DONE already exists for job $index; use status/receive validation"
    }
    $name = (($RunName.ToLowerInvariant() -replace '[^a-z0-9-]','-').Trim('-'))
    if ($name.Length -gt 50) { $name = $name.Substring(0,50).TrimEnd('-') }
    $instance = "$name-pf$index"
    $describeResult = Invoke-M43A4GcloudProcess `
        -Arguments @(
            'compute','instances','describe',$instance,
            '--zone',$Zone,'--project',$ProjectId,'--format=json'
        ) `
        -TimeoutSeconds 60 -Label "describe Attempt08 instance $instance"
    if (-not $describeResult.timed_out -and $describeResult.exit_code -eq 0) {
        throw "Instance exists: $instance"
    }
    $describeMessage = ([string]$describeResult.stdout) + ([string]$describeResult.stderr)
    if ($describeResult.timed_out -or
        $describeMessage -notmatch '(?i)not found|was not found|does not exist|404') {
        throw "Unable to prove instance absence: $instance"
    }
    $metadata = @(
        "RUN_NAME=$RunName", "PROJECT_ID=$ProjectId", "BUCKET=$Bucket", "JOB_INDEX=$index",
        "SOURCE_URI=$Prefix/source/source.zip", "SOURCE_SHA256=$($manifestObject.source_zip_sha256)",
        "MANIFEST_SHA256=$(Hash $Manifest)", "AUTHORIZATION_SHA256=$(Hash $Authorization)",
        "STARTUP_SHA256=$($manifestObject.startup_sha256)", "SELF_DELETE=$selfDelete"
    ) -join ','
    $createResult = Invoke-M43A4GcloudProcess `
        -Arguments @(
            'compute','instances','create',$instance,
            '--project',$ProjectId,'--zone',$Zone,
            '--machine-type','c4-highmem-4',
            '--provisioning-model','SPOT',
            '--instance-termination-action','DELETE',
            '--maintenance-policy','TERMINATE','--no-restart-on-failure',
            '--image','debian-12-bookworm-v20260609','--image-project','debian-cloud',
            '--boot-disk-size','50GB','--boot-disk-type','hyperdisk-balanced',
            '--metadata',$metadata,
            '--metadata-from-file',"startup-script=$(Join-Path $RunDir 'startup_hu_m43_attempt08_preflight.sh')",
            '--scopes','cloud-platform'
        ) `
        -TimeoutSeconds 600 -Label "create Attempt08 instance $instance"
    if ($createResult.timed_out -or $createResult.exit_code -ne 0) {
        throw "Failed to create $instance"
    }
}
Write-Host "Started $($selected.Count) bounded c4-highmem-4 Spot jobs."
