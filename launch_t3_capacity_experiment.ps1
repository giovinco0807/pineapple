param(
    [Parameter(Position = 0)]
    [ValidateSet("upload", "create", "status", "gcs-status", "download", "delete", "all")]
    [string]$Action = "status",
    [string]$Project = "ofc-solver-485418",
    [string]$Bucket = "gs://pokerhu-ofc-solver-485418-training",
    [string]$RunId = "t3-capacity-20260603",
    [string]$VmName = "ofc-t3-capacity-20260603",
    [string]$Zone = "us-central1-a",
    [string]$MachineType = "n1-standard-8",
    [string]$GpuType = "nvidia-tesla-t4",
    [int]$GpuCount = 1,
    [string]$DiskSize = "120GB",
    [int]$Epochs = 40,
    [int]$MaxSeconds = 0,
    [int]$BatchSize = 1024,
    [int]$RankingBatches = 128,
    [int]$UploadInterval = 300,
    [string]$DownloadDir = "ai\data\hybrid_t1t2_active_20260531\gcp_t3_capacity_20260603",
    [switch]$KeepVM
)

$ErrorActionPreference = "Stop"

$LocalRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$StartupScript = Join-Path $LocalRoot "gcp_t3_capacity_startup.sh"
$CodeArchiveName = "ofc_t3_capacity_code.tar.gz"
$DataArchiveName = "ofc_t3_capacity_data.tar.gz"
$CodeArchiveObject = "$Bucket/runs/$RunId/code/$CodeArchiveName"
$DataArchiveObject = "$Bucket/runs/$RunId/code/$DataArchiveName"
$GcsResultRoot = "$Bucket/runs/$RunId/results"

function Invoke-Gcloud {
    & gcloud @args
}

function New-CodeArchive {
    $tarPath = Join-Path $env:TEMP ("ofc_t3_capacity_code_{0}.tar.gz" -f $RunId)
    Push-Location $LocalRoot
    try {
        if (Test-Path $tarPath) {
            Remove-Item -LiteralPath $tarPath -Force
        }
        tar --exclude="__pycache__" -czf $tarPath `
            ai/__init__.py `
            ai/engine `
            ai/models/__init__.py `
            ai/models/action_value_reranker.py `
            ai/models/networks.py `
            ai/training/__init__.py `
            ai/training/train_action_value_reranker.py `
            ai/training/convert_recursive_teacher_to_reranker.py `
            ai/training/convert_action_value_teacher.py `
            ai/training/convert_mc_teacher.py `
            ai/tutor/__init__.py `
            ai/tutor/run_t3_data_capacity_experiment.py `
            ai/tutor/evaluate_teacher_model.py `
            ai/tutor/benchmark_t2_t3_value_model.py `
            ai/tutor/exact_late.py `
            ai/tutor/run_t2_exact_oracle.py
    } finally {
        Pop-Location
    }
    return $tarPath
}

function New-DataArchive {
    $tarPath = Join-Path $env:TEMP ("ofc_t3_capacity_data_{0}.tar.gz" -f $RunId)
    Push-Location $LocalRoot
    try {
        if (Test-Path $tarPath) {
            Remove-Item -LiteralPath $tarPath -Force
        }
        tar -czf $tarPath `
            ai/data/t0_proxy_holdout_20260529/exact_late_reranker_t3t4_10000_20260529 `
            ai/models/candidate_runs/tutor-route10-t3-exact10000-adapter64-20260529/model/action_value_best.pt `
            ai/data/tutor_eval_holdout_20260523/teacher_labels_t3_exact_500.jsonl `
            ai/data/hybrid_t1t2_active_20260531/oracle_first_next2_rows40_49_cap100_20260603/inputs/t2_input.jsonl `
            ai/data/hybrid_t1t2_active_20260531/oracle_first_next2_rows40_49_cap100_20260603/t2_caps/t2_oracle_cap100_limit10.jsonl
    } finally {
        Pop-Location
    }
    return $tarPath
}

function Upload-CodeAndData {
    if (-not (Test-Path $StartupScript)) {
        throw "startup script not found: $StartupScript"
    }
    $codeTar = New-CodeArchive
    $dataTar = New-DataArchive
    Write-Host "Uploading code archive to $CodeArchiveObject"
    Invoke-Gcloud storage cp $codeTar $CodeArchiveObject --project=$Project
    Write-Host "Uploading data archive to $DataArchiveObject"
    Invoke-Gcloud storage cp $dataTar $DataArchiveObject --project=$Project
}

function Create-VM {
    $selfDelete = if ($KeepVM) { "false" } else { "true" }
    $metadataItems = @(
        "bucket=$Bucket",
        "run-id=$RunId",
        "code-archive=$CodeArchiveObject",
        "data-archive=$DataArchiveObject",
        "epochs=$Epochs",
        "max-seconds=$MaxSeconds",
        "batch-size=$BatchSize",
        "ranking-batches=$RankingBatches",
        "upload-interval=$UploadInterval",
        "self-delete=$selfDelete"
    )
    $metadata = $metadataItems -join ","

    Write-Host "Creating $VmName in $Zone"
    Invoke-Gcloud compute instances create $VmName `
        --project=$Project `
        --zone=$Zone `
        --machine-type=$MachineType `
        --image-family="pytorch-2-9-cu129-ubuntu-2204-nvidia-580" `
        --image-project="deeplearning-platform-release" `
        --boot-disk-size=$DiskSize `
        --boot-disk-type="pd-balanced" `
        --accelerator="type=$GpuType,count=$GpuCount" `
        --maintenance-policy=TERMINATE `
        --provisioning-model=SPOT `
        --instance-termination-action=STOP `
        --scopes="default,storage-rw" `
        --labels="purpose=ofc-t3-capacity,run=$RunId" `
        --metadata=$metadata `
        --metadata-from-file="startup-script=$StartupScript" `
        --quiet
}

function Show-Status {
    Invoke-Gcloud compute instances list `
        --project=$Project `
        --filter="labels.run=$RunId" `
        --format="table(name,zone.basename(),machineType.basename(),status,creationTimestamp,labels.purpose)"
}

function Show-GcsStatus {
    Write-Host "GCS result root: $GcsResultRoot/"
    Invoke-Gcloud storage ls --recursive "$GcsResultRoot/" --project=$Project
}

function Download-Results {
    $localOut = Join-Path $LocalRoot $DownloadDir
    New-Item -ItemType Directory -Force -Path $localOut | Out-Null
    Invoke-Gcloud storage cp --recursive "$GcsResultRoot/" $localOut --project=$Project
    Write-Host "Downloaded results to $localOut"
}

function Delete-VM {
    Invoke-Gcloud compute instances delete $VmName --project=$Project --zone=$Zone --quiet
}

switch ($Action) {
    "upload" { Upload-CodeAndData }
    "create" { Create-VM }
    "status" { Show-Status }
    "gcs-status" { Show-GcsStatus }
    "download" { Download-Results }
    "delete" { Delete-VM }
    "all" {
        Upload-CodeAndData
        Create-VM
        Show-Status
        Write-Host "Results will appear under $GcsResultRoot/"
    }
}
