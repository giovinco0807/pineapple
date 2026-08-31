param(
    [Parameter(Position = 0)]
    [ValidateSet("upload", "create", "status", "gcs-status", "restart-startup", "delete", "all")]
    [string]$Action = "status",
    [string]$Project = "ofc-solver-485418",
    [string]$Bucket = "gs://pokerhu-ofc-solver-485418-training",
    [string]$RunId = "recursive-t0t3-s128-b5-c2-p16-tr32-20260520",
    [string]$Prefix = "ofc-rec-t0t3-20260520",
    [int]$StartIndex = 0,
    [int]$NumVMs = 8,
    [string]$MachineType = "n2-highcpu-16",
    [string]$DiskSize = "50GB",
    [int]$HandsPerVM = 100,
    [int]$SeedBase = 202605200,
    [int]$Sims = 128,
    [int]$Beam = 5,
    [int]$ChildSims = 2,
    [int]$PoolSize = 16,
    [int]$TraceRollouts = 1,
    [int]$TraceSims = 32,
    [int]$TracePoolSize = 6,
    [int]$RayonThreads = 16,
    [int]$UploadInterval = 300,
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
    $Zones = $ZonesCsv.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ }
}

$LocalRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$StartupScript = Join-Path $LocalRoot "gcp_recursive_teacher_startup.sh"
$ArchiveName = "ofc_recursive_teacher_code.tar.gz"
$CodeArchiveObject = "$Bucket/runs/$RunId/code/$ArchiveName"

function Get-VmName([int]$Index) {
    return ("{0}-{1:D2}" -f $Prefix, $Index)
}

function Get-VmZone([int]$Index) {
    return $Zones[($Index - $StartIndex) % $Zones.Count]
}

function New-CodeArchive {
    $tarPath = Join-Path $env:TEMP ("ofc_recursive_teacher_code_{0}.tar.gz" -f $RunId)
    Push-Location $LocalRoot
    try {
        if (Test-Path $tarPath) {
            Remove-Item -LiteralPath $tarPath -Force
        }
        tar --exclude="ai/rust_solver/target" -czf $tarPath `
            ai/__init__.py `
            ai/prob_engine_wrapper.py `
            ai/engine `
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
    gcloud storage cp $tarPath $CodeArchiveObject --project=$Project
    Write-Host "Uploaded: $CodeArchiveObject"
}

function Create-VMs {
    Write-Host "Creating $NumVMs spot VMs for run $RunId"
    Write-Host "Each VM: $MachineType, start_index=$StartIndex, hands=$HandsPerVM, sims=$Sims, trace_sims=$TraceSims"
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
            "pool-size=$PoolSize",
            "trace-rollouts=$TraceRollouts",
            "trace-sims=$TraceSims",
            "trace-pool-size=$TracePoolSize",
            "rayon-threads=$RayonThreads",
            "upload-interval=$UploadInterval",
            "self-delete=$selfDelete"
        ) -join ","

        Write-Host "Creating $name in $zone"
        gcloud compute instances create $name `
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
            --labels="purpose=ofc-recursive-teacher,run=$RunId" `
            --metadata=$metadata `
            --metadata-from-file="startup-script=$StartupScript" `
            --quiet
    }
}

function Show-Status {
    gcloud compute instances list `
        --project=$Project `
        --filter="labels.run=$RunId" `
        --format="table(name,zone.basename(),machineType.basename(),status,creationTimestamp,labels.purpose)"
}

function Show-GcsStatus {
    Write-Host "GCS result root: $Bucket/runs/$RunId/results/"
    gcloud storage ls --recursive "$Bucket/runs/$RunId/results/" --project=$Project
}

function Restart-Startup {
    for ($i = $StartIndex; $i -lt ($StartIndex + $NumVMs); $i++) {
        $name = Get-VmName $i
        $zone = Get-VmZone $i
        Write-Host "Updating startup script and resetting $name in $zone"
        gcloud compute instances add-metadata $name `
            --project=$Project `
            --zone=$zone `
            --metadata-from-file="startup-script=$StartupScript" `
            --quiet
        gcloud compute instances reset $name --project=$Project --zone=$zone --quiet
    }
}

function Delete-VMs {
    for ($i = $StartIndex; $i -lt ($StartIndex + $NumVMs); $i++) {
        $name = Get-VmName $i
        $zone = Get-VmZone $i
        Write-Host "Deleting $name in $zone"
        gcloud compute instances delete $name --project=$Project --zone=$zone --quiet
    }
}

switch ($Action) {
    "upload" { Upload-Code }
    "create" { Create-VMs }
    "status" { Show-Status }
    "gcs-status" { Show-GcsStatus }
    "restart-startup" { Restart-Startup }
    "delete" { Delete-VMs }
    "all" {
        Upload-Code
        Create-VMs
        Show-Status
        Write-Host "Results will appear under $Bucket/runs/$RunId/results/"
    }
}
