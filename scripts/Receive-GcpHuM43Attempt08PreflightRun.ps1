param(
    [Parameter(Mandatory)][string]$RunName,
    [string]$ProjectId = 'ofc-solver-485418',
    [string]$Bucket = 'pokerhu-ofc-solver-485418-training',
    [string]$RunDir,
    [string]$OutputDir
)
$ErrorActionPreference='Stop'; Set-StrictMode -Version Latest
$RepoRoot=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if($RunName -notmatch '^[a-z0-9][a-z0-9-]{2,126}[a-z0-9]$'){throw 'Unsafe RunName'}
if (-not $RunDir) { $RunDir=Join-Path $RepoRoot "outputs/gcp_runs/$RunName" }
$ExpectedRunDir=[IO.Path]::GetFullPath((Join-Path $RepoRoot "outputs/gcp_runs/$RunName"))
$RunDir=[IO.Path]::GetFullPath($RunDir)
if($RunDir -cne $ExpectedRunDir){throw 'RunDir must be the exact repository run directory'}
$FrozenScript=Join-Path $RunDir 'package_src/scripts/Receive-GcpHuM43Attempt08PreflightRun.ps1'
if(-not (Test-Path -LiteralPath $FrozenScript -PathType Leaf) -or
   (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash -cne
   (Get-FileHash -LiteralPath $FrozenScript -Algorithm SHA256).Hash){
    throw 'Receive-GcpHuM43Attempt08PreflightRun.ps1 differs from its frozen packaged copy'
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
$FrozenPreflightSource=Join-Path $RunDir 'package_src/preflight_source/teacher.jsonl'
if(-not (Test-Path -LiteralPath $FrozenPreflightSource -PathType Leaf)){
    throw 'Frozen packaged preflight source is missing'
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
$ExpectedOutputDir=[IO.Path]::GetFullPath(
    (Join-Path $RepoRoot "outputs/hu_joint_policy/m43_attempt08_preflight/$RunName")
)
if (-not $OutputDir) { $OutputDir=$ExpectedOutputDir }
$OutputDir=[IO.Path]::GetFullPath($OutputDir)
if($OutputDir -cne $ExpectedOutputDir){
    throw "OutputDir must be the exact external preflight result directory: $ExpectedOutputDir"
}
$RunPrefix=$RunDir.TrimEnd(
    [IO.Path]::DirectorySeparatorChar,
    [IO.Path]::AltDirectorySeparatorChar
) + [IO.Path]::DirectorySeparatorChar
if($OutputDir.StartsWith($RunPrefix,[StringComparison]::OrdinalIgnoreCase)){
    throw 'OutputDir must remain outside the protected Spot run directory'
}
$Mirror=Join-Path $RunDir 'receive_mirror'
$ExpectedMirror=[IO.Path]::GetFullPath((Join-Path $ExpectedRunDir 'receive_mirror'))
if([IO.Path]::GetFullPath($Mirror) -cne $ExpectedMirror){throw 'Unsafe receive mirror path'}
$MirrorPrefix=$ExpectedMirror.TrimEnd(
    [IO.Path]::DirectorySeparatorChar,
    [IO.Path]::AltDirectorySeparatorChar
) + [IO.Path]::DirectorySeparatorChar
if($OutputDir -eq $ExpectedMirror -or
   $OutputDir.StartsWith($MirrorPrefix,[StringComparison]::OrdinalIgnoreCase)){
    throw 'OutputDir must remain outside the protected receive mirror'
}
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
        throw "output_prefix escaped receive mirror: $prefix"
    }
    $resolved
}
# Phase 1 downloads DONE only. No proof is opened until every DONE validates.
foreach($s in $schedule){
    $dir=Resolve-JobMirror $s; New-Item -ItemType Directory -Path $dir | Out-Null
    Copy-M43A4RemoteFileExact `
        -Uri "gs://$Bucket/runs/$RunName/results/$($s.output_prefix)/DONE.json" `
        -Destination (Join-Path $dir 'DONE.json') -ProjectId $ProjectId
}
& python -m ofc_regular.hu_m43_attempt08_preflight_spot status --run-dir $RunDir `
    --authorization (Join-Path $RunDir 'spot_authorization.json') `
    --local-evidence (Join-Path $RunDir 'local_evidence.json') --jobs-root $Mirror
if($LASTEXITCODE -ne 0){throw 'All-five DONE validation failed'}
# Phase 2 may now fetch proof/support artifacts.
foreach($s in $schedule){
    $dir=Resolve-JobMirror $s
    foreach($name in @('proof.json','checkpoint.json','heartbeat.json','summary.json','run.log','boot_image_evidence.json')){
        Copy-M43A4RemoteFileExact `
            -Uri "gs://$Bucket/runs/$RunName/results/$($s.output_prefix)/$name" `
            -Destination (Join-Path $dir $name) -ProjectId $ProjectId
    }
}
& python -m ofc_regular.hu_m43_attempt08_preflight_spot receive --run-dir $RunDir `
    --authorization (Join-Path $RunDir 'spot_authorization.json') `
    --local-evidence (Join-Path $RunDir 'local_evidence.json') `
    --jobs-root $Mirror --output-dir $OutputDir --source $FrozenPreflightSource
if($LASTEXITCODE -ne 0){throw 'Attempt08 receive/aggregate/finalize failed'}
