param(
    [Parameter(Position = 0)]
    [ValidateSet("upload", "create", "status", "gcs-status", "delete", "all")]
    [string]$Action = "status",
    [string]$Project = "ofc-solver-485418",
    [string]$Bucket = "gs://pokerhu-ofc-solver-485418-training",
    [string]$RunId = "joint-t0-aa-k20-trips45-top30-s100-b10-20260521",
    [string]$Prefix = "ofc-joint-t0-20260521",
    [int]$StartIndex = 0,
    [int]$NumVMs = 8,
    [string]$MachineType = "n2-highcpu-16",
    [string]$DiskSize = "50GB",
    [int]$HandsPerVM = 1,
    [int]$SeedBase = 2026052100,
    [int]$Sims = 100,
    [int]$Beam = 10,
    [int]$ChildSims = 2,
    [int]$T0PoolSize = 30,
    [int]$TraceRollouts = 1,
    [int]$TraceSims = 32,
    [string]$TracePoolSizes = "0;15;10",
    [double]$FlAnyWeight = 0.0,
    [double]$FlQqWeight = 0.0,
    [double]$FlKkWeight = 20.0,
    [double]$FlAaWeight = 30.0,
    [double]$FlTripsWeight = 45.0,
    [double]$EvWeight = 0.02,
    [double]$BustWeight = 0.0,
    [int]$RayonThreads = 16,
    [int]$ProbEngineTimeout = 600,
    [int]$UploadInterval = 180,
    [string]$ZonesCsv = "",
    [switch]$KeepVMs
)

$ErrorActionPreference = "Stop"

$Zones = @(
    "us-west1-b",
    "us-west1-c",
    "us-central1-a",
    "us-central1-b",
    "us-central1-c",
    "us-east1-b",
    "us-east1-c"
)
if ($ZonesCsv) {
    $Zones = @($ZonesCsv.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ })
}

$LocalRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$StartupScript = Join-Path $LocalRoot "gcp_joint_t0_teacher_startup.sh"
$ArchiveName = "ofc_joint_t0_teacher_code.tar.gz"
$CodeArchiveObject = "$Bucket/runs/$RunId/code/$ArchiveName"

function Get-VmName([int]$Index) {
    return ("{0}-{1:D2}" -f $Prefix, $Index)
}

function Get-VmZone([int]$Index) {
    return $Zones[($Index - $StartIndex) % $Zones.Count]
}

function New-CodeArchive {
    $tarPath = Join-Path $env:TEMP ("ofc_joint_t0_teacher_code_{0}.tar.gz" -f $RunId)
    Push-Location $LocalRoot
    try {
        if (Test-Path $tarPath) {
            Remove-Item -LiteralPath $tarPath -Force
        }
        tar --exclude="ai/rust_solver/target" -czf $tarPath `
            ai/__init__.py `
            ai/prob_engine_wrapper.py `
            ai/engine `
            ai/training/generate_joint_t0_recursive_teacher.py `
            ai/training/generate_recursive_t0_random.py `
            ai/rust_solver
    } finally {
        Pop-Location
    }
    return $tarPath
}

function Upload-Code {
    if (-not (Test-Path $StartupScript)) {
        throw "startup script not found: $StartupScript"
    }
    $tarPath = New-CodeArchive
    Write-Host "Uploading code archive to $CodeArchiveObject"
    gcloud.cmd storage cp $tarPath $CodeArchiveObject --project=$Project
    Write-Host "Uploaded: $CodeArchiveObject"
}

function Create-VMs {
    Write-Host "Creating $NumVMs spot VMs for run $RunId"
    Write-Host "Each VM: $MachineType, start_index=$StartIndex, hands=$HandsPerVM, sims=$Sims, beam=$Beam, t0_pool=$T0PoolSize"
    Write-Host "Objective weights: AA=$FlAaWeight KK=$FlKkWeight Trips=$FlTripsWeight QQ=$FlQqWeight any=$FlAnyWeight ev=$EvWeight bust=$BustWeight"
    $selfDelete = if ($KeepVMs) { "false" } else { "true" }
    for ($i = $StartIndex; $i -lt ($StartIndex + $NumVMs); $i++) {
        $name = Get-VmName $i
        $zone = Get-VmZone $i
        $seed = $SeedBase + $i
        $metadata = @(
            "bucket=$Bucket",
            "run-id=$RunId",
            "worker-id=$i",
            "code-archive=$CodeArchiveObject",
            "hands=$HandsPerVM",
            "seed=$seed",
            "sims=$Sims",
            "beam=$Beam",
            "child-sims=$ChildSims",
            "t0-pool-size=$T0PoolSize",
            "trace-rollouts=$TraceRollouts",
            "trace-sims=$TraceSims",
            "trace-pool-sizes=$TracePoolSizes",
            "fl-any-weight=$FlAnyWeight",
            "fl-qq-weight=$FlQqWeight",
            "fl-kk-weight=$FlKkWeight",
            "fl-aa-weight=$FlAaWeight",
            "fl-trips-weight=$FlTripsWeight",
            "ev-weight=$EvWeight",
            "bust-weight=$BustWeight",
            "rayon-threads=$RayonThreads",
            "prob-engine-timeout=$ProbEngineTimeout",
            "upload-interval=$UploadInterval",
            "self-delete=$selfDelete"
        ) -join ","

        Write-Host "Creating $name in $zone"
        gcloud.cmd compute instances create $name `
            --project=$Project `
            --zone=$zone `
            --machine-type=$MachineType `
            --image-family="ubuntu-2204-lts" `
            --image-project="ubuntu-os-cloud" `
            --boot-disk-size=$DiskSize `
            --boot-disk-type="pd-standard" `
            --provisioning-model=SPOT `
            --instance-termination-action=STOP `
            --scopes="default,storage-rw" `
            --labels="purpose=ofc-joint-t0-teacher,run=$RunId" `
            --metadata=$metadata `
            --metadata-from-file="startup-script=$StartupScript" `
            --quiet
    }
}

function Show-Status {
    gcloud.cmd compute instances list `
        --project=$Project `
        --filter="labels.run=$RunId" `
        --format="table(name,zone.basename(),machineType.basename(),status,creationTimestamp,labels.purpose)"
}

function Show-GcsStatus {
    Write-Host "GCS result root: $Bucket/runs/$RunId/results/"
    gcloud.cmd storage ls --recursive "$Bucket/runs/$RunId/results/" --project=$Project
}

function Delete-VMs {
    for ($i = $StartIndex; $i -lt ($StartIndex + $NumVMs); $i++) {
        $name = Get-VmName $i
        $zone = Get-VmZone $i
        Write-Host "Deleting $name in $zone"
        gcloud.cmd compute instances delete $name --project=$Project --zone=$zone --quiet
    }
}

switch ($Action) {
    "upload" { Upload-Code }
    "create" { Create-VMs }
    "status" { Show-Status }
    "gcs-status" { Show-GcsStatus }
    "delete" { Delete-VMs }
    "all" {
        Upload-Code
        Create-VMs
        Show-Status
        Write-Host "Results will appear under $Bucket/runs/$RunId/results/"
    }
}
