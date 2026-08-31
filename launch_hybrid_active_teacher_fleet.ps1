param(
    [Parameter(Position = 0)]
    [ValidateSet("upload", "create", "status", "gcs-status", "delete", "all")]
    [string]$Action = "status",
    [string]$Project = "ofc-solver-485418",
    [string]$Bucket = "gs://pokerhu-ofc-solver-485418-training",
    [string]$RunId = "hybrid-t1t2-mc300-6400-20260531",
    [string]$Prefix = "ofc-hybrid-t1t2-20260531",
    [string]$ShardDir = "ai\data\hybrid_t1t2_active_20260531\gcp_mc300_shards_16",
    [int]$StartIndex = 0,
    [int]$NumVMs = 16,
    [string]$MachineType = "n2-highcpu-16",
    [string]$DiskSize = "50GB",
    [int]$Sims = 300,
    [string]$Turns = "1,2",
    [string]$McTurns = "1,2",
    [int]$RayonThreads = 16,
    [int]$UploadInterval = 180,
    [int]$BatchTimeout = 86400,
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
    "us-east1-c",
    "us-east4-a"
)
if ($ZonesCsv) {
    $Zones = @($ZonesCsv.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ })
}

$LocalRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$StartupScript = Join-Path $LocalRoot "gcp_hybrid_active_teacher_startup.sh"
$ArchiveName = "ofc_hybrid_active_teacher_code.tar.gz"
$CodeArchiveObject = "$Bucket/runs/$RunId/code/$ArchiveName"
$ShardRoot = Join-Path $LocalRoot $ShardDir
$GcsInputRoot = "$Bucket/runs/$RunId/inputs"

function Get-VmName([int]$Index) {
    return ("{0}-{1:D2}" -f $Prefix, $Index)
}

function Get-VmZone([int]$Index) {
    return $Zones[($Index - $StartIndex) % $Zones.Count]
}

function Get-LocalShardPath([int]$Index) {
    return Join-Path $ShardRoot ("targets_shard_{0:D2}.jsonl" -f $Index)
}

function Get-GcsShardObject([int]$Index) {
    $suffix = "{0:D2}" -f $Index
    return "$GcsInputRoot/targets_shard_$suffix.jsonl"
}

function New-CodeArchive {
    $tarPath = Join-Path $env:TEMP ("ofc_hybrid_active_teacher_code_{0}.tar.gz" -f $RunId)
    Push-Location $LocalRoot
    try {
        if (Test-Path $tarPath) {
            Remove-Item -LiteralPath $tarPath -Force
        }
        tar --exclude="ai/rust_solver/target" -czf $tarPath `
            ai/__init__.py `
            ai/prob_engine_wrapper.py `
            ai/training/__init__.py `
            ai/training/generate_active_teacher.py `
            ai/rust_solver
    } finally {
        Pop-Location
    }
    return $tarPath
}

function Upload-CodeAndInputs {
    if (-not (Test-Path $StartupScript)) {
        throw "startup script not found: $StartupScript"
    }
    if (-not (Test-Path $ShardRoot)) {
        throw "shard dir not found: $ShardRoot"
    }
    $tarPath = New-CodeArchive
    Write-Host "Uploading code archive to $CodeArchiveObject"
    gcloud.cmd storage cp $tarPath $CodeArchiveObject --project=$Project

    Write-Host "Uploading target shards to $GcsInputRoot/"
    for ($i = $StartIndex; $i -lt ($StartIndex + $NumVMs); $i++) {
        $localShard = Get-LocalShardPath $i
        if (-not (Test-Path $localShard)) {
            throw "target shard not found: $localShard"
        }
        gcloud.cmd storage cp $localShard (Get-GcsShardObject $i) --project=$Project
    }
    $manifest = Join-Path $ShardRoot "manifest.json"
    if (Test-Path $manifest) {
        gcloud.cmd storage cp $manifest "$GcsInputRoot/manifest.json" --project=$Project
    }
    Write-Host "Uploaded code and shards for run $RunId"
}

function Create-VMs {
    Write-Host "Creating $NumVMs spot VMs for run $RunId"
    Write-Host "Each VM: $MachineType, sims=$Sims, shard_dir=$ShardRoot"
    $selfDelete = if ($KeepVMs) { "false" } else { "true" }
    $metadataTurns = $Turns -replace ",", ";"
    $metadataMcTurns = $McTurns -replace ",", ";"
    for ($i = $StartIndex; $i -lt ($StartIndex + $NumVMs); $i++) {
        $name = Get-VmName $i
        $zone = Get-VmZone $i
        $metadataItems = @(
            "bucket=$Bucket",
            "run-id=$RunId",
            "worker-id=$i",
            "code-archive=$CodeArchiveObject",
            "target-shard=$(Get-GcsShardObject $i)",
            "sims=$Sims",
            "turns=$metadataTurns",
            "mc-turns=$metadataMcTurns",
            "rayon-threads=$RayonThreads",
            "upload-interval=$UploadInterval",
            "batch-timeout=$BatchTimeout",
            "self-delete=$selfDelete"
        )
        $metadata = $metadataItems -join ","

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
            --labels="purpose=ofc-hybrid-active-teacher,run=$RunId" `
            --metadata=$metadata `
            --metadata-from-file="startup-script=$StartupScript" `
            --quiet
        if ($LASTEXITCODE -ne 0) {
            throw "failed to create $name in $zone"
        }
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
    "upload" { Upload-CodeAndInputs }
    "create" { Create-VMs }
    "status" { Show-Status }
    "gcs-status" { Show-GcsStatus }
    "delete" { Delete-VMs }
    "all" {
        Upload-CodeAndInputs
        Create-VMs
        Show-Status
        Write-Host "Results will appear under $Bucket/runs/$RunId/results/"
    }
}
