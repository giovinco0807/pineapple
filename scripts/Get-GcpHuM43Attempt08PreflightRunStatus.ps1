param(
    [Parameter(Mandatory)][string]$RunName,
    [string]$ProjectId = 'ofc-solver-485418',
    [string]$Bucket = 'pokerhu-ofc-solver-485418-training',
    [string]$RunDir
)
$ErrorActionPreference='Stop'; Set-StrictMode -Version Latest
$RepoRoot=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if($RunName -notmatch '^[a-z0-9][a-z0-9-]{2,126}[a-z0-9]$'){throw 'Unsafe RunName'}
if (-not $RunDir) { $RunDir=Join-Path $RepoRoot "outputs/gcp_runs/$RunName" }
$ExpectedRunDir=[IO.Path]::GetFullPath((Join-Path $RepoRoot "outputs/gcp_runs/$RunName"))
$RunDir=[IO.Path]::GetFullPath($RunDir)
if($RunDir -cne $ExpectedRunDir){throw 'RunDir must be the exact repository run directory'}
$FrozenScript=Join-Path $RunDir 'package_src/scripts/Get-GcpHuM43Attempt08PreflightRunStatus.ps1'
if(-not (Test-Path -LiteralPath $FrozenScript -PathType Leaf) -or
   (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash -cne
   (Get-FileHash -LiteralPath $FrozenScript -Algorithm SHA256).Hash){
    throw 'Get-GcpHuM43Attempt08PreflightRunStatus.ps1 differs from its frozen packaged copy'
}
$FrozenCommon08=Join-Path $RunDir 'package_src/scripts/HuM43Attempt08Spot.Common.ps1'
$FrozenCommon04=Join-Path $RunDir 'package_src/scripts/HuM43Attempt04Spot.Common.ps1'
foreach($binding in @(
    @{
        outer=Join-Path $RepoRoot 'scripts/HuM43Attempt08Spot.Common.ps1'
        frozen=$FrozenCommon08
        label='HuM43Attempt08Spot.Common.ps1'
    },
    @{
        outer=Join-Path $RepoRoot 'scripts/HuM43Attempt04Spot.Common.ps1'
        frozen=$FrozenCommon04
        label='HuM43Attempt04Spot.Common.ps1'
    }
)){
    if(-not (Test-Path -LiteralPath $binding.frozen -PathType Leaf) -or
       (Get-FileHash -LiteralPath $binding.outer -Algorithm SHA256).Hash -cne
       (Get-FileHash -LiteralPath $binding.frozen -Algorithm SHA256).Hash){
        throw "$($binding.label) differs from its frozen packaged copy"
    }
}
. $FrozenCommon08
$FrozenSourceRoot=Join-Path $RunDir 'package_src/src'
if(-not (Test-Path -LiteralPath $FrozenSourceRoot -PathType Container)){
    throw 'Frozen package source is missing'
}
$env:PYTHONPATH=$FrozenSourceRoot
$env:PYTHONDONTWRITEBYTECODE='1'
& python -m ofc_regular.hu_m43_attempt08_preflight_spot validate-launch `
    --manifest (Join-Path $RunDir 'manifest.json') `
    --authorization (Join-Path $RunDir 'spot_authorization.json') `
    --local-evidence (Join-Path $RunDir 'local_evidence.json')
if($LASTEXITCODE -ne 0){throw 'Package/schedule/authorization validation failed'}
$manifestObject=Get-Content -LiteralPath (Join-Path $RunDir 'manifest.json') -Raw | ConvertFrom-Json
if([string]$manifestObject.run_name -cne $RunName){throw 'Package run_name changed'}
$Mirror=Join-Path $RunDir 'done_status_mirror'
$ExpectedMirror=[IO.Path]::GetFullPath((Join-Path $ExpectedRunDir 'done_status_mirror'))
if([IO.Path]::GetFullPath($Mirror) -cne $ExpectedMirror){throw 'Unsafe status mirror path'}
if (Test-Path $Mirror) { Remove-Item -LiteralPath $Mirror -Recurse -Force }
New-Item -ItemType Directory -Path $Mirror | Out-Null
$schedule=Get-Content -LiteralPath (Join-Path $RunDir 'shards_manifest.jsonl') | ForEach-Object { $_ | ConvertFrom-Json }
function Resolve-JobMirror([object]$Shard) {
    $prefix=[string]$Shard.output_prefix
    if($prefix -notmatch '^job_00[0-4]_(root0_batch_a|root0_batch_b|root0_scalar|root1_batch|root2_batch)$'){
        throw "Unsafe output_prefix: $prefix"
    }
    $resolved=[IO.Path]::GetFullPath((Join-Path $Mirror $prefix))
    if([IO.Path]::GetDirectoryName($resolved) -cne [IO.Path]::GetFullPath($Mirror)){
        throw "output_prefix escaped status mirror: $prefix"
    }
    $resolved
}
foreach($s in $schedule){
    $dir=Resolve-JobMirror $s; New-Item -ItemType Directory -Path $dir | Out-Null
    $uri="gs://$Bucket/runs/$RunName/results/$($s.output_prefix)/DONE.json"
    if(Test-M43A4GcsObject -Uri $uri -ProjectId $ProjectId){
        Copy-M43A4RemoteFileExact `
            -Uri $uri -Destination (Join-Path $dir 'DONE.json') -ProjectId $ProjectId
    }
}
& python -m ofc_regular.hu_m43_attempt08_preflight_spot status `
    --run-dir $RunDir --authorization (Join-Path $RunDir 'spot_authorization.json') `
    --local-evidence (Join-Path $RunDir 'local_evidence.json') --jobs-root $Mirror
if ($LASTEXITCODE -ne 0) { throw 'DONE-only status validation failed' }
