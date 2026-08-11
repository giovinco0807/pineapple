param(
    [string]$ProjectId = "ofc-solver-485418",
    [string]$Bucket = "pokerhu-ofc-solver-485418-training",
    [string]$RunName = ("regular-hu-t0-stage1-all232-mc4-" + (Get-Date -Format "yyyyMMdd-HHmmss")),
    [int]$TotalSamples = 100,
    [int]$ShardSamples = 2,
    [int]$RecordSkipBase = 0,
    [int]$RecordSkipStride = 0,
    [switch]$FastSkipRecords,
    [int]$BaseSeed = 2026103001,
    [int]$SeedStride = 1000000,
    [int]$VmCount = 20,
    [string]$MachineType = "c4-highmem-2",
    [int]$CpuThreads = 1,
    [int]$FutureSamples = 4,
    [int]$MaxActions = 0,
    [string]$CandidateModel = "",
    [string[]]$CandidateModels = @(),
    [int]$CandidateTopk = 0,
    [int]$CandidateUnionCap = 0,
    [ValidateSet("min_rank", "rank_sum", "reciprocal_rank_sum", "mean_score", "max_score", "mean_z_score", "max_z_score")]
    [string]$CandidateUnionMode = "min_rank",
    [string[]]$Seats = @("first", "second"),
    [string[]]$Zones = @(
        "asia-northeast1-a",
        "asia-northeast1-b",
        "asia-northeast1-c",
        "us-central1-a",
        "us-central1-b",
        "us-central1-c",
        "us-east1-b",
        "us-east1-c",
        "us-west1-b",
        "us-west1-c"
    ),
    [string[]]$StartShards = @(),
    [switch]$CollectTopkLog,
    [switch]$CreateInstances,
    [switch]$SkipExistingInstances,
    [switch]$NoSpot,
    [switch]$NoSelfDelete,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$runner = Join-Path $PSScriptRoot "Start-GcpHuTurn1PilotRun.ps1"
if (-not (Test-Path -LiteralPath $runner)) {
    throw "Generic pilot runner not found: $runner"
}

$arguments = @{
    ProjectId = $ProjectId
    Bucket = $Bucket
    RunName = $RunName
    TotalSamples = $TotalSamples
    ShardSamples = $ShardSamples
    RecordSkipBase = $RecordSkipBase
    RecordSkipStride = $RecordSkipStride
    FastSkipRecords = [bool]$FastSkipRecords
    BaseSeed = $BaseSeed
    SeedStride = $SeedStride
    VmCount = $VmCount
    MachineType = $MachineType
    CpuThreads = $CpuThreads
    Zones = $Zones
    StartShards = $StartShards
    FutureSamples = $FutureSamples
    MaxActions = $MaxActions
    CandidateModel = $CandidateModel
    CandidateModels = $CandidateModels
    CandidateTopk = $CandidateTopk
    CandidateUnionCap = $CandidateUnionCap
    CandidateUnionMode = $CandidateUnionMode
    OpeningLookaheadSamples = 1
    Profile = "stage18_p1"
    OpponentProfile = "stage18_p1"
    SourceBucket = ("natural_terminal_mc{0}" -f $FutureSamples)
    Seats = $Seats
    TeacherModule = "ofc_regular.hu_turn0_teacher_pilot"
    PhaseName = "hu_t0_stage1_pilot"
    AdditionalRequiredModels = @(
        "models/hu_turn1_stage18_first1801_mc32_hgb_abs_aug3.pkl",
        "models/hu_turn1_stage18_highmc_safe_selector_a_meta_only.pkl"
    )
    CollectTopkLog = [bool]$CollectTopkLog
    CreateInstances = [bool]$CreateInstances
    SkipExistingInstances = [bool]$SkipExistingInstances
    NoSpot = [bool]$NoSpot
    NoSelfDelete = [bool]$NoSelfDelete
    DryRun = [bool]$DryRun
}

& $runner @arguments
if ($LASTEXITCODE -ne 0) {
    throw "HU T0 GCP pilot runner failed with exit code $LASTEXITCODE"
}
