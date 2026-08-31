param(
    [Parameter(Position = 0)]
    [ValidateSet("upload", "create", "status", "gcs-status", "download", "delete", "all")]
    [string]$Action = "status",
    [string]$Project = "ofc-solver-485418",
    [string]$Bucket = "gs://pokerhu-ofc-solver-485418-training",
    [string]$RunId = "branch-t0t2-top10-100k-20260605",
    [string]$Prefix = "ofc-branch-t3-20260605",
    [int]$StartIndex = 0,
    [int]$NumVMs = 10,
    [int]$RootsPerWorker = 6,
    [int]$MaxOutputsPerWorker = 10000,
    [int]$T0TopK = 10,
    [int]$RegularTopK = 10,
    [string]$Position = "both",
    [int]$SeedBase = 2026060500,
    [string]$MachineType = "n2-highcpu-16",
    [string]$DiskSize = "90GB",
    [int]$RayonThreads = 16,
    [int]$UploadInterval = 180,
    [int]$BatchSize = 512,
    [string]$Device = "cpu",
    [string]$ZonesCsv = "",
    [string]$DownloadDir = "D:\ofc-pineapple-data\branch_t0t2_top10_100k_20260605\gcp_results",
    [switch]$KeepVMs
)

$ErrorActionPreference = "Stop"

$Zones = @(
    "us-west1-b",
    "us-west1-c",
    "us-central1-a",
    "us-central1-b",
    "us-central1-c",
    "us-central1-f",
    "us-east1-b",
    "us-east1-c",
    "us-east4-a",
    "us-east4-b"
)
if ($ZonesCsv) {
    $Zones = @($ZonesCsv.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ })
}

$LocalRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$StartupScript = Join-Path $LocalRoot "gcp_branch_expanded_t3_startup.sh"
$ArchiveName = "ofc_branch_expanded_t3_code.tar.gz"
$CodeArchiveObject = "$Bucket/runs/$RunId/code/$ArchiveName"
$GcsResultRoot = "$Bucket/runs/$RunId/results"

function Get-VmName([int]$Index) {
    return ("{0}-{1:D2}" -f $Prefix, $Index)
}

function Get-VmZone([int]$Index) {
    return $Zones[($Index - $StartIndex) % $Zones.Count]
}

function Invoke-Gcloud {
    & gcloud.cmd @args
    if ($LASTEXITCODE -ne 0) {
        throw "gcloud failed: $($args -join ' ')"
    }
}

function New-CodeArchive {
    $tarPath = Join-Path $env:TEMP ("ofc_branch_expanded_t3_code_{0}.tar.gz" -f $RunId)
    Push-Location $LocalRoot
    try {
        if (Test-Path $tarPath) {
            Remove-Item -LiteralPath $tarPath -Force
        }
        tar --exclude="ai/rust_solver/target" --exclude="__pycache__" -czf $tarPath `
            ai/__init__.py `
            ai/engine `
            ai/models/__init__.py `
            ai/models/action_value_reranker.py `
            ai/models/networks.py `
            ai/models/candidate_runs/tutor-route10-t0top128-route3-mc30-1000-ft-20260525/model/action_value_best.pt `
            ai/models/candidate_runs/t1-runtime112-top40regret-x20-plus-currenthard15-x20-ft-20260604/model/action_value_best.pt `
            ai/models/candidate_runs/t2-mix-broad80k-hard121-x10-ft-20260603/action_value_best.pt `
            ai/models/candidate_runs/t3-jokerfix-20k-bb-20260605/action_value_best.pt `
            ai/training/__init__.py `
            ai/training/action_feature_encoding.py `
            ai/training/convert_action_value_teacher.py `
            ai/training/convert_mc_teacher.py `
            ai/tutor/__init__.py `
            ai/tutor/build_branch_expanded_t3_targets.py `
            ai/tutor/convert_rust_t3_exact_teacher.py `
            ai/config/fl_ev.json `
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
    Invoke-Gcloud storage cp $tarPath $CodeArchiveObject --project=$Project
    Write-Host "Uploaded code archive for run $RunId"
}

function Create-VMs {
    $selfDelete = if ($KeepVMs) { "false" } else { "true" }
    Write-Host "Creating $NumVMs spot VMs for $RunId"
    Write-Host "Plan: $NumVMs workers x $MaxOutputsPerWorker T3 positions = $($NumVMs * $MaxOutputsPerWorker) target positions"
    for ($i = $StartIndex; $i -lt ($StartIndex + $NumVMs); $i++) {
        $name = Get-VmName $i
        $zone = Get-VmZone $i
        $metadataItems = @(
            "bucket=$Bucket",
            "run-id=$RunId",
            "worker-id=$i",
            "code-archive=$CodeArchiveObject",
            "roots=$RootsPerWorker",
            "max-outputs=$MaxOutputsPerWorker",
            "t0-top-k=$T0TopK",
            "regular-top-k=$RegularTopK",
            "position=$Position",
            "seed-base=$SeedBase",
            "batch-size=$BatchSize",
            "device=$Device",
            "rayon-threads=$RayonThreads",
            "upload-interval=$UploadInterval",
            "self-delete=$selfDelete"
        )
        $metadata = $metadataItems -join ","

        Write-Host "Creating $name in $zone"
        Invoke-Gcloud compute instances create $name `
            --project=$Project `
            --zone=$zone `
            --machine-type=$MachineType `
            --image-family="ubuntu-2204-lts" `
            --image-project="ubuntu-os-cloud" `
            --boot-disk-size=$DiskSize `
            --boot-disk-type="pd-balanced" `
            --provisioning-model=SPOT `
            --instance-termination-action=STOP `
            --scopes="default,storage-rw" `
            --labels="purpose=ofc-branch-t3,run=$RunId" `
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
    Write-Host "GCS result root: $GcsResultRoot/"
    gcloud.cmd storage ls --recursive "$GcsResultRoot/" --project=$Project
}

function Download-Results {
    New-Item -ItemType Directory -Force -Path $DownloadDir | Out-Null
    Invoke-Gcloud storage cp --recursive "$GcsResultRoot/" $DownloadDir --project=$Project
    Write-Host "Downloaded results to $DownloadDir"
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
    "download" { Download-Results }
    "delete" { Delete-VMs }
    "all" {
        Upload-Code
        Create-VMs
        Show-Status
        Write-Host "Results will appear under $GcsResultRoot/"
    }
}
