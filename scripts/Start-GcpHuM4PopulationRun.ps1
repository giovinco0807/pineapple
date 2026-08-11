param(
    [Parameter(Mandatory = $true)][string]$ModelPath,
    [Parameter(Mandatory = $true)][string]$TrainingManifestPath,
    [Parameter(Mandatory = $true)][string]$DataContractPath,
    [Parameter(Mandatory = $true)][string]$FreezeManifestPath,
    [Parameter(Mandatory = $true)][string]$LockedReceiptPath,
    [Parameter(Mandatory = $true)][string]$ConsumptionMarkerPath,
    [string]$PlanPath = "configs/hu_joint_policy_m43_population.json",
    [string]$ProjectId = "ofc-solver-485418",
    [string]$Bucket = "pokerhu-ofc-solver-485418-training",
    [string]$RunName = ("regular-hu-m43-population-" + (Get-Date -Format "yyyyMMdd-HHmmss")),
    [string]$MachineType = "c4-standard-4",
    [string[]]$FallbackMachineTypes = @("n2-standard-4", "e2-standard-4"),
    [string[]]$Zones = @("asia-northeast1-b", "asia-northeast1-c", "us-central1-a"),
    [int]$BootDiskGb = 30,
    [int]$SyncIntervalSeconds = 60,
    [string[]]$StartShards = @(),
    [switch]$CreateInstances,
    [switch]$ResumeExisting,
    [switch]$SkipExistingInstances,
    [switch]$NoSelfDelete,
    [switch]$PackageOnly,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$RequiredModels = @(
    "models/opening_stage7_torch_wide.pt",
    "models/turn1_stage6_torch_wide.pt",
    "models/turn2_stage8.pkl",
    "models/turn3_stage6.pkl",
    "models/hu_turn2_stage8b_safe_lcb196_20k_reference_override_cached_rank_wide.pt",
    "models/hu_turn3_stage7_reference_override_cached_rank_wide.pt",
    "models/hu_turn3_stage3_mc32_500k_plus_m8_12_f128_100k_w2_cached_rank_wide.pt",
    "models/hu_turn1_stage18_first1801_mc32_hgb_abs_aug3.pkl",
    "models/hu_turn1_stage18_highmc_safe_selector_a_meta_only.pkl",
    "models/hu_turn0_stage3_top60_mc32_1000_hgb_baseline-delta_sew1_aug3.pkl",
    "models/hu_turn0_stage19_safe_selector_highmc_candidate_delta_plus_meta.pkl"
)
$RequiredNativeBinaries = @(
    "target/release/libofc_stage3_feature_encoder.so",
    "target/release/libofc_hu_m3_engine.so"
)

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Assert-Sha256Text {
    param($Value, [Parameter(Mandatory = $true)][string]$Label)
    if ($Value -isnot [string] -or $Value -cnotmatch '^[0-9a-f]{64}$') {
        throw "$Label must be a lowercase SHA-256 digest"
    }
}

function Convert-RequiredJsonInteger {
    param($Value, [Parameter(Mandatory = $true)][string]$Label)
    if ($null -eq $Value -or $Value -is [bool] -or $Value -is [string] -or
        $Value -is [single] -or $Value -is [double] -or $Value -is [decimal]) {
        throw "$Label must be a JSON integer"
    }
    try { return [long]$Value } catch { throw "$Label must be a JSON integer" }
}

function Assert-LinuxX8664Elf {
    param([Parameter(Mandatory = $true)][string]$Path)
    $stream=[System.IO.File]::OpenRead($Path)
    try{
        $header=New-Object byte[] 20
        if($stream.Read($header,0,$header.Length) -ne $header.Length -or
            $header[0] -ne 0x7f -or $header[1] -ne 0x45 -or
            $header[2] -ne 0x4c -or $header[3] -ne 0x46 -or
            $header[4] -ne 2 -or $header[5] -ne 1 -or
            $header[16] -ne 0x03 -or $header[17] -ne 0x00 -or
            $header[18] -ne 0x3e -or $header[19] -ne 0x00){
            throw "Population native dependency is not Linux x86_64 ELF: $Path"
        }
    }finally{$stream.Dispose()}
}

function Write-Utf8NoBom {
    param([Parameter(Mandatory = $true)][string]$Path, [Parameter(Mandatory = $true)][string]$Text)
    [System.IO.File]::WriteAllText($Path, $Text, [System.Text.UTF8Encoding]::new($false))
}

function New-ZipWithForwardSlashes {
    param([Parameter(Mandatory = $true)][string]$SourceDir, [Parameter(Mandatory = $true)][string]$DestinationPath)
    Add-Type -AssemblyName System.IO.Compression
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    if (Test-Path -LiteralPath $DestinationPath) { Remove-Item -LiteralPath $DestinationPath -Force }
    $sourceRoot = (Resolve-Path -LiteralPath $SourceDir).Path.TrimEnd('\', '/')
    $prefixLength = $sourceRoot.Length + 1
    $zip = [System.IO.Compression.ZipFile]::Open($DestinationPath, [System.IO.Compression.ZipArchiveMode]::Create)
    try {
        Get-ChildItem -LiteralPath $sourceRoot -Recurse -File | Sort-Object FullName | ForEach-Object {
            $entryName = $_.FullName.Substring($prefixLength) -replace '\\', '/'
            [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
                $zip, $_.FullName, $entryName, [System.IO.Compression.CompressionLevel]::Optimal
            ) | Out-Null
        }
    }
    finally { $zip.Dispose() }
}

function Invoke-Gcloud {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { $output = & gcloud @Arguments 2>&1; $code = $LASTEXITCODE }
    finally { $ErrorActionPreference = $old }
    if ($code -ne 0) { throw "gcloud failed ($code): gcloud $($Arguments -join ' ')`n$(@($output) -join [Environment]::NewLine)" }
    return @($output)
}

function Test-GcsObject {
    param([Parameter(Mandatory = $true)][string]$Uri)
    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { $output = & gcloud storage objects describe $Uri --format=json 2>&1; $code = $LASTEXITCODE }
    finally { $ErrorActionPreference = $old }
    if ($code -eq 0) { return $true }
    if ((@($output) -join "`n") -match '(?i)not found|does not exist|No URLs matched|404') { return $false }
    throw "Could not inspect immutable GCS object: $Uri`n$(@($output) -join [Environment]::NewLine)"
}

function Convert-ToVmPrefix {
    param([Parameter(Mandatory = $true)][string]$Value)
    $name = ($Value.ToLowerInvariant() -replace '[^a-z0-9-]', '-').Trim('-')
    if (-not $name) { throw "RunName does not produce a VM prefix" }
    if ($name -notmatch '^[a-z]') { $name = "m4p-$name" }
    if ($name.Length -gt 50) { $name = $name.Substring(0, 50).TrimEnd('-') }
    return $name
}

if ($RunName -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$') { throw "RunName contains unsafe characters" }
if ($BootDiskGb -lt 20) { throw "BootDiskGb must be at least 20" }
if ($SyncIntervalSeconds -lt 15) { throw "SyncIntervalSeconds must be at least 15" }
if (-not $Zones -or $Zones.Count -eq 0) { throw "At least one zone is required" }
if ($PackageOnly -and $CreateInstances) { throw "PackageOnly and CreateInstances are mutually exclusive" }
if ($DryRun -and $CreateInstances) { throw "DryRun never creates cloud instances" }
if ($ResumeExisting -and $PackageOnly) { throw "ResumeExisting cannot repackage a frozen run" }

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
function Resolve-RepoPath([string]$Value) {
    $candidate = if ([System.IO.Path]::IsPathRooted($Value)) { $Value } else { Join-Path $repoRoot $Value }
    return (Resolve-Path -LiteralPath $candidate).Path
}

$runDir = Join-Path $repoRoot "outputs/gcp_runs/$RunName"
$packageDir = Join-Path $runDir "package"
$sourceArchivePath = Join-Path $runDir "ofc_regular_hu_m4_population_source.zip"
$manifestPath = Join-Path $runDir "population_run_manifest.json"
$shardManifestPath = Join-Path $runDir "population_shards.jsonl"
$startupPath = Join-Path $runDir "startup_hu_m4_population.sh"
$gcsPrefix = "gs://$Bucket/runs/$RunName"
$manifestUri = "$gcsPrefix/source/population_run_manifest.json"
$sourceUri = "$gcsPrefix/source/ofc_regular_hu_m4_population_source.zip"
$startupUri = "$gcsPrefix/source/startup_hu_m4_population.sh"
$shardsUri = "$gcsPrefix/source/population_shards.jsonl"
$vmPrefix = Convert-ToVmPrefix $RunName

if ($ResumeExisting) {
    foreach ($path in @($manifestPath, $sourceArchivePath, $startupPath, $shardManifestPath)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Frozen population run input is missing: $path" }
    }
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    $requiredResumeHashes = [ordered]@{
        "source.sha256" = $manifest.source.sha256
        "startup.sha256" = $manifest.startup.sha256
        "shards.sha256" = $manifest.shards.sha256
        "population_plan.sha256" = $manifest.population_plan.sha256
        "runtime.model_sha256" = $manifest.runtime.model_sha256
        "runtime.training_manifest_sha256" = $manifest.runtime.training_manifest_sha256
        "runtime.data_contract_sha256" = $manifest.runtime.data_contract_sha256
        "runtime.freeze_manifest_sha256" = $manifest.runtime.freeze_manifest_sha256
        "runtime.locked_receipt_sha256" = $manifest.runtime.locked_receipt_sha256
        "runtime.consumption_marker_sha256" = $manifest.runtime.consumption_marker_sha256
        "launch_preflight.model_sha256" = $manifest.launch_preflight.model_sha256
        "launch_preflight.training_manifest_sha256" = $manifest.launch_preflight.training_manifest_sha256
        "launch_preflight.data_contract_file_sha256" = $manifest.launch_preflight.data_contract_file_sha256
        "launch_preflight.population_plan_file_sha256" = $manifest.launch_preflight.population_plan_file_sha256
        "launch_preflight.freeze_manifest_canonical_sha256" = $manifest.launch_preflight.freeze_manifest_canonical_sha256
        "launch_preflight.locked_receipt_canonical_sha256" = $manifest.launch_preflight.locked_receipt_canonical_sha256
        "launch_preflight.consumption_marker_canonical_sha256" = $manifest.launch_preflight.consumption_marker_canonical_sha256
        "launch_preflight.freeze_manifest_file_sha256" = $manifest.launch_preflight.freeze_manifest_file_sha256
        "launch_preflight.locked_receipt_file_sha256" = $manifest.launch_preflight.locked_receipt_file_sha256
        "launch_preflight.consumption_marker_sha256" = $manifest.launch_preflight.consumption_marker_sha256
    }
    foreach ($hashField in $requiredResumeHashes.GetEnumerator()) {
        Assert-Sha256Text -Value $hashField.Value -Label $hashField.Key
    }
    $teacherOverlap = Convert-RequiredJsonInteger $manifest.launch_preflight.teacher_overlap_count "launch_preflight.teacher_overlap_count"
    $priorOverlap = Convert-RequiredJsonInteger $manifest.launch_preflight.prior_population_overlap_count "launch_preflight.prior_population_overlap_count"
    $teacherChecked = Convert-RequiredJsonInteger $manifest.launch_preflight.teacher_hand_seeds_checked "launch_preflight.teacher_hand_seeds_checked"
    $populationChecked = Convert-RequiredJsonInteger $manifest.launch_preflight.population_hand_seeds_checked "launch_preflight.population_hand_seeds_checked"
    $planPairedSeeds = Convert-RequiredJsonInteger $manifest.population_plan.paired_seeds "population_plan.paired_seeds"
    $manifestShardCount = Convert-RequiredJsonInteger $manifest.shards.count "shards.count"
    if ($manifest.schema -ne "hu_m4_population_spot_manifest_v1" -or
        $manifest.run_name -ne $RunName -or $manifest.project_id -ne $ProjectId -or
        $manifest.bucket -ne $Bucket -or $manifest.no_runtime_activation -ne $true -or
        $manifest.current_profile_mutated -ne $false -or
        $manifest.population_plan.uri -ne "$gcsPrefix/source/population_plan.json" -or
        $manifest.launch_preflight.schema -ne "hu_m43_population_launch_preflight_v1" -or
        $manifest.launch_preflight.status -ne "pass" -or
        $teacherOverlap -ne 0 -or $priorOverlap -ne 0 -or
        $teacherChecked -le 0 -or $populationChecked -le 0 -or
        $planPairedSeeds -le 0 -or $manifestShardCount -le 0 -or
        $manifest.launch_preflight.model_sha256 -ne $manifest.runtime.model_sha256 -or
        $manifest.launch_preflight.training_manifest_sha256 -ne $manifest.runtime.training_manifest_sha256 -or
        $manifest.launch_preflight.population_plan_file_sha256 -ne $manifest.population_plan.sha256 -or
        $manifest.launch_preflight.freeze_manifest_file_sha256 -ne $manifest.runtime.freeze_manifest_sha256 -or
        $manifest.launch_preflight.locked_receipt_file_sha256 -ne $manifest.runtime.locked_receipt_sha256 -or
        $manifest.launch_preflight.consumption_marker_sha256 -ne $manifest.runtime.consumption_marker_sha256 -or
        $populationChecked -ne $planPairedSeeds) {
        throw "Frozen population run identity mismatch"
    }
    if ((Get-Sha256 $sourceArchivePath) -ne $manifest.source.sha256 -or
        (Get-Sha256 $startupPath) -ne $manifest.startup.sha256 -or
        (Get-Sha256 $shardManifestPath) -ne $manifest.shards.sha256) {
        throw "Frozen population run local hash chain is broken"
    }
    if (-not $DryRun) {
        foreach ($uri in @($manifestUri, $sourceUri, $startupUri, $shardsUri, $manifest.population_plan.uri)) {
            if (-not (Test-GcsObject $uri)) { throw "Frozen remote population input is missing: $uri" }
        }
        $remoteManifestPath = [System.IO.Path]::GetTempFileName()
        $remotePlanPath = [System.IO.Path]::GetTempFileName()
        try {
            Invoke-Gcloud @("storage", "cp", $manifestUri, $remoteManifestPath, "--project", $ProjectId) | Out-Null
            if ((Get-Sha256 $remoteManifestPath) -ne (Get-Sha256 $manifestPath)) {
                throw "Frozen remote population manifest disagrees with the local resume manifest"
            }
            Invoke-Gcloud @("storage", "cp", $manifest.population_plan.uri, $remotePlanPath, "--project", $ProjectId) | Out-Null
            if ((Get-Sha256 $remotePlanPath) -ne $manifest.population_plan.sha256) {
                throw "Frozen remote population plan disagrees with the resume manifest"
            }
        }
        finally {
            Remove-Item -LiteralPath $remoteManifestPath, $remotePlanPath -Force -ErrorAction SilentlyContinue
        }
    }
}
else {
    if (Test-Path -LiteralPath $runDir) { throw "Population run directory already exists and is immutable: $runDir" }
    $resolved = [ordered]@{
        plan = Resolve-RepoPath $PlanPath
        model = Resolve-RepoPath $ModelPath
        training_manifest = Resolve-RepoPath $TrainingManifestPath
        data_contract = Resolve-RepoPath $DataContractPath
        freeze_manifest = Resolve-RepoPath $FreezeManifestPath
        locked_receipt = Resolve-RepoPath $LockedReceiptPath
        consumption_marker = Resolve-RepoPath $ConsumptionMarkerPath
    }
    $plan = Get-Content -LiteralPath $resolved.plan -Raw | ConvertFrom-Json
    $training = Get-Content -LiteralPath $resolved.training_manifest -Raw | ConvertFrom-Json
    $contract = Get-Content -LiteralPath $resolved.data_contract -Raw | ConvertFrom-Json
    $freeze = Get-Content -LiteralPath $resolved.freeze_manifest -Raw | ConvertFrom-Json
    $receipt = Get-Content -LiteralPath $resolved.locked_receipt -Raw | ConvertFrom-Json
    $lifecyclePreflight = @'
import hashlib
import json
import math
import sys
from pathlib import Path

from ofc_regular.hu_m43_pilot_contract import (
    canonical_manifest_sha256,
    validate_locked_holdout_receipt,
    validate_model_threshold_freeze,
)


def read_mapping(path: Path, label: str):
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def mapping(value, label: str):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def sequence(value, label: str):
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def integer(value, label: str, *, positive: bool = False):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


(
    model_path,
    training_path,
    contract_path,
    freeze_path,
    receipt_path,
    marker_path,
    plan_path,
) = (Path(value).resolve() for value in sys.argv[1:])
training = read_mapping(training_path, "training manifest")
contract = read_mapping(contract_path, "data contract")
freeze = read_mapping(freeze_path, "freeze manifest")
receipt = read_mapping(receipt_path, "locked receipt")
marker = read_mapping(marker_path, "consumption marker")
plan = read_mapping(plan_path, "population plan")

# Reuse the authoritative lifecycle validators instead of duplicating a
# partial PowerShell field check. They validate the canonical contract,
# freeze, one-shot receipt, threshold, and all no-activation declarations.
validate_model_threshold_freeze(freeze, data_contract=contract)
validate_locked_holdout_receipt(
    receipt,
    freeze_manifest=freeze,
    data_contract=contract,
)

model_sha = file_sha256(model_path)
training_sha = file_sha256(training_path)
marker_sha = file_sha256(marker_path)
if model_sha != freeze.get("model_sha256"):
    raise ValueError("model bytes disagree with freeze")
if training_sha != freeze.get("training_manifest_sha256"):
    raise ValueError("training-manifest bytes disagree with freeze")
if marker_sha != receipt.get("consumption_marker_sha256"):
    raise ValueError("marker bytes disagree with locked receipt")

freeze_canonical_sha = canonical_manifest_sha256(freeze)
receipt_canonical_sha = canonical_manifest_sha256(receipt)
marker_canonical_sha = canonical_manifest_sha256(marker)
expected_marker_keys = {
    "schema",
    "status",
    "freeze_manifest_sha256",
    "data_contract_sha256",
    "model_sha256",
    "locked_identity_sha256",
    "locked_teacher_shards_sha256",
    "evaluation_pass_count",
}
if set(marker) != expected_marker_keys:
    raise ValueError("consumption marker key set changed")
marker_checks = {
    "schema": marker.get("schema")
    == "hu_m43_locked_holdout_consumption_marker_v1",
    "status": marker.get("status") == "claimed_before_locked_content_read",
    "freeze": marker.get("freeze_manifest_sha256") == freeze_canonical_sha,
    "contract": marker.get("data_contract_sha256")
    == contract.get("contract_sha256"),
    "model": marker.get("model_sha256") == model_sha,
    "locked_identity": marker.get("locked_identity_sha256")
    == mapping(mapping(contract.get("splits"), "contract.splits").get("locked_holdout"), "contract locked split").get("identity_sha256"),
    "locked_shards": marker.get("locked_teacher_shards_sha256")
    == mapping(
        mapping(
            mapping(contract.get("teacher_shards"), "contract.teacher_shards").get("splits"),
            "contract.teacher_shards.splits",
        ).get("locked_holdout"),
        "contract locked shard binding",
    ).get("ordered_shards_sha256"),
    "pass_count": marker.get("evaluation_pass_count") == 1,
}
if not all(marker_checks.values()):
    raise ValueError(f"consumption marker content mismatch: {marker_checks}")

# The freeze binds the training file bytes; verify that the bound training
# manifest is itself the expected M4.3 pre-holdout artifact.
runtime_lock = mapping(training.get("runtime_lock"), "training.runtime_lock")
formula = mapping(training.get("action_score_formula"), "training.action_score_formula")
calibration = mapping(training.get("calibration"), "training.calibration")
binding = mapping(training.get("m43_data_contract"), "training.m43_data_contract")
if formula.get("mode") != "baseline_paired_delta_risk_ensemble_v3":
    raise ValueError("training action-score mode is not M4.3")
if training.get("locked_holdout") != {"status": "not_evaluated_pre_freeze"}:
    raise ValueError("training manifest opened locked holdout before freeze")
if set(mapping(training.get("inputs"), "training.inputs")) != {"train", "calibration"}:
    raise ValueError("training manifest input roles changed")
if runtime_lock.get("candidate_model_sha256") != model_sha or runtime_lock.get("safety_model_sha256") != model_sha:
    raise ValueError("training runtime lock disagrees with model bytes")
if not math.isclose(
    float(runtime_lock.get("safety_threshold")),
    float(freeze.get("frozen_threshold")),
    rel_tol=0.0,
    abs_tol=1.0e-12,
):
    raise ValueError("training runtime threshold disagrees with freeze")
if calibration.get("threshold_selection_source") != "calibration.threshold_lock":
    raise ValueError("training threshold source changed")
if calibration.get("safety_calibrator_sources") != [
    "train_oof",
    "calibration.safety_fit",
]:
    raise ValueError("training safety-calibrator sources changed")
for key, expected in {
    "contract_sha256": contract.get("contract_sha256"),
    "plan_sha256": contract.get("plan_sha256"),
    "base_audit_sha256": contract.get("base_audit_sha256"),
    "teacher_shards_all_splits_sha256": contract.get(
        "teacher_shards_all_splits_sha256"
    ),
}.items():
    if binding.get(key) != expected:
        raise ValueError(f"training/data-contract binding mismatch: {key}")

if plan.get("schema") != "hu_m43_population_acceptance_plan_v1" or plan.get("status") != "frozen_before_population_evaluation":
    raise ValueError("population plan is not frozen M4.3")
if plan.get("opponents") != [
    "stage19_p0",
    "stage9f_p2",
    "stage7_m5_r10",
    "random_exact_final",
] or plan.get("paired_seat_swap") is not True:
    raise ValueError("population opponent/seat-swap plan changed")
seed = integer(plan.get("seed"), "plan.seed", positive=True)
stride = integer(plan.get("seed_stride"), "plan.seed_stride", positive=True)
count = integer(
    plan.get("paired_seeds_per_opponent"),
    "plan.paired_seeds_per_opponent",
    positive=True,
)
shards = integer(plan.get("shards"), "plan.shards", positive=True)
per_shard = integer(
    plan.get("paired_seeds_per_shard"),
    "plan.paired_seeds_per_shard",
    positive=True,
)
if shards * per_shard != count:
    raise ValueError("population shard grid is incomplete")
if integer(plan.get("minimum_valid_overrides"), "plan.minimum_valid_overrides") < 300:
    raise ValueError("population override gate was lowered")
expected_guards = {
    "current_profile_changed",
    "runtime_policy_activated",
    "full_replacement_enabled",
    "threshold_changed_after_lock",
}
guards = mapping(plan.get("activation_guards"), "plan.activation_guards")
if set(guards) != expected_guards or any(value is not False for value in guards.values()):
    raise ValueError("population activation guards changed")
sizing = mapping(plan.get("sizing_rule"), "plan.sizing_rule")
if sizing.get("fixed_before_realized_population_labels") is not True or sizing.get("optional_stopping_or_posthoc_extension_allowed") is not False:
    raise ValueError("population sizing/optional-stopping contract changed")

planned_seeds = {seed + index * stride for index in range(count)}
base_audit = mapping(contract.get("base_audit"), "contract.base_audit")
if base_audit.get("schema") != "hu_m4_t1_second_data_audit_v1" or base_audit.get("status") != "pass":
    raise ValueError("data-contract base audit is not passing")
teacher_seeds = set()
for split_name, split in mapping(base_audit.get("splits"), "base_audit.splits").items():
    for shard_index, shard in enumerate(
        sequence(mapping(split, f"base_audit split {split_name}").get("shards"), f"base_audit {split_name}.shards")
    ):
        for hand_seed in sequence(
            mapping(shard, f"base_audit {split_name} shard {shard_index}").get("hand_seeds"),
            f"base_audit {split_name} shard {shard_index}.hand_seeds",
        ):
            teacher_seeds.add(
                integer(hand_seed, f"base_audit {split_name} hand seed", positive=True)
            )
if not teacher_seeds:
    raise ValueError("base audit exposes no teacher hand seeds")
teacher_overlap = planned_seeds & teacher_seeds
if teacher_overlap:
    raise ValueError(f"population/teacher seed overlap: {sorted(teacher_overlap)[:5]}")

freshness = mapping(plan.get("freshness"), "plan.freshness")
if freshness.get("exclude_all_m4_m41_m42_m43_teacher_hand_seeds") is not True or freshness.get("exclude_prior_population_smoke_seeds") is not True or freshness.get("planned_overlap_count_at_freeze") != 0:
    raise ValueError("population freshness declarations changed")
prior_schedules = sequence(
    freshness.get("excluded_population_schedules"),
    "plan.freshness.excluded_population_schedules",
)
if not prior_schedules:
    raise ValueError("prior population schedules are missing")
prior_overlap = set()
for index, raw_schedule in enumerate(prior_schedules):
    schedule = mapping(raw_schedule, f"prior population schedule {index}")
    prior_seed = integer(schedule.get("seed"), f"prior schedule {index}.seed", positive=True)
    prior_stride = integer(
        schedule.get("seed_stride"),
        f"prior schedule {index}.seed_stride",
        positive=True,
    )
    prior_count = integer(
        schedule.get("paired_seeds"),
        f"prior schedule {index}.paired_seeds",
        positive=True,
    )
    prior_overlap.update(
        planned_seeds
        & {prior_seed + item * prior_stride for item in range(prior_count)}
    )
if prior_overlap:
    raise ValueError(f"population/prior schedule seed overlap: {sorted(prior_overlap)[:5]}")

print(
    json.dumps(
        {
            "schema": "hu_m43_population_launch_preflight_v1",
            "status": "pass",
            "model_sha256": model_sha,
            "training_manifest_sha256": training_sha,
            "freeze_manifest_canonical_sha256": freeze_canonical_sha,
            "locked_receipt_canonical_sha256": receipt_canonical_sha,
            "consumption_marker_canonical_sha256": marker_canonical_sha,
            "freeze_manifest_file_sha256": file_sha256(freeze_path),
            "locked_receipt_file_sha256": file_sha256(receipt_path),
            "consumption_marker_sha256": marker_sha,
            "data_contract_file_sha256": file_sha256(contract_path),
            "population_plan_file_sha256": file_sha256(plan_path),
            "teacher_hand_seeds_checked": len(teacher_seeds),
            "population_hand_seeds_checked": len(planned_seeds),
            "teacher_overlap_count": 0,
            "prior_population_overlap_count": 0,
        },
        sort_keys=True,
    )
)
'@
    $oldPythonPath = $env:PYTHONPATH
    $oldPreference = $ErrorActionPreference
    $env:PYTHONPATH = Join-Path $repoRoot "src"
    $ErrorActionPreference = "Continue"
    try {
        $preflightRaw = @($lifecyclePreflight | & python - $resolved.model $resolved.training_manifest $resolved.data_contract $resolved.freeze_manifest $resolved.locked_receipt $resolved.consumption_marker $resolved.plan 2>&1)
        $preflightCode = $LASTEXITCODE
    }
    finally {
        $env:PYTHONPATH = $oldPythonPath
        $ErrorActionPreference = $oldPreference
    }
    if ($preflightCode -ne 0) {
        throw "M4.3 population launch preflight failed ($preflightCode):`n$(@($preflightRaw) -join [Environment]::NewLine)"
    }
    $preflight = (@($preflightRaw) -join "`n") | ConvertFrom-Json
    if ($preflight.schema -ne "hu_m43_population_launch_preflight_v1" -or $preflight.status -ne "pass" -or [int]$preflight.teacher_overlap_count -ne 0 -or [int]$preflight.prior_population_overlap_count -ne 0) {
        throw "M4.3 population launch preflight returned an invalid receipt"
    }
    if ($plan.schema -ne "hu_m43_population_acceptance_plan_v1" -or
        $plan.status -ne "frozen_before_population_evaluation" -or
        [int]$plan.minimum_valid_overrides -lt 300 -or
        (@($plan.opponents) -join ',') -ne 'stage19_p0,stage9f_p2,stage7_m5_r10,random_exact_final' -or
        $plan.activation_guards.current_profile_changed -ne $false) {
        throw "Population plan is not the frozen M4.3 acceptance plan"
    }
    if ([int]$plan.shards * [int]$plan.paired_seeds_per_shard -ne [int]$plan.paired_seeds_per_opponent) {
        throw "Population shard plan does not cover its seed grid"
    }
    $modelSha = Get-Sha256 $resolved.model
    $trainingSha = Get-Sha256 $resolved.training_manifest
    $markerSha = Get-Sha256 $resolved.consumption_marker
    if ($freeze.schema -ne "hu_m43_model_threshold_freeze_v1" -or
        $freeze.status -ne "model_and_threshold_frozen_locked_unopened" -or
        $freeze.model_sha256 -ne $modelSha -or
        $freeze.training_manifest_sha256 -ne $trainingSha -or
        $freeze.calibration_status -ne "go" -or
        $freeze.safety_enabled -ne $true -or
        $freeze.current_profile_resolved -ne $false -or
        $freeze.runtime_policy_activated -ne $false -or
        [int]$freeze.minimum_population_valid_overrides -lt 300 -or
        $receipt.schema -ne "hu_m43_locked_holdout_receipt_v1" -or
        $receipt.status -ne "evaluated_once_diagnostic_only_no_activation" -or
        $receipt.model_sha256 -ne $modelSha -or
        $receipt.consumption_marker_sha256 -ne $markerSha -or
        [int]$receipt.evaluation_pass_count -ne 1 -or
        $receipt.threshold_search_performed -ne $false -or
        $receipt.model_selection_performed -ne $false -or
        $receipt.current_profile_resolved -ne $false -or
        $receipt.requires_fresh_population_acceptance -ne $true -or
        [int]$receipt.minimum_population_valid_overrides -lt 300 -or
        $training.runtime_lock.candidate_model_sha256 -ne $modelSha -or
        $training.runtime_lock.safety_model_sha256 -ne $modelSha -or
        $training.promotion_status -ne "candidate_for_realized_ev_evaluation" -or
        $training.calibration.status -ne "go" -or
        [int]$training.calibration.selected_metrics.fires -lt 10 -or
        [int]$training.calibration.constraints.minimum_fires -lt 10 -or
        $contract.contract_sha256 -ne $freeze.data_contract_sha256 -or
        $contract.contract_sha256 -ne $receipt.data_contract_sha256) {
        throw "M4.3 model/freeze/one-shot lifecycle hash chain is not launchable"
    }
    $oldPythonPath=$env:PYTHONPATH
    $env:PYTHONPATH=(Join-Path $repoRoot "src")
    try {
        & python -c "import json,sys; from ofc_regular.hu_m43_pilot_contract import validate_model_threshold_freeze,validate_locked_holdout_receipt; from ofc_regular.merge_hu_m4_population_shards import load_population_plan; c=json.load(open(sys.argv[1],encoding='utf-8-sig')); f=json.load(open(sys.argv[2],encoding='utf-8-sig')); r=json.load(open(sys.argv[3],encoding='utf-8-sig')); validate_model_threshold_freeze(f,data_contract=c); validate_locked_holdout_receipt(r,freeze_manifest=f,data_contract=c); load_population_plan(sys.argv[4])" $resolved.data_contract $resolved.freeze_manifest $resolved.locked_receipt $resolved.plan
        if($LASTEXITCODE -ne 0){throw "Python lifecycle/plan validator rejected the population launch"}
    }
    finally {$env:PYTHONPATH=$oldPythonPath}

    # Do not leave a partial immutable run directory when any launch preflight
    # rejects the lifecycle or seed contract.
    New-Item -ItemType Directory -Path $packageDir -Force | Out-Null
    foreach ($item in @("pyproject.toml", "Cargo.toml", "Cargo.lock", "src", "configs")) {
        Copy-Item -LiteralPath (Join-Path $repoRoot $item) -Destination $packageDir -Recurse
    }
    $rustDir = Join-Path $packageDir "rust"
    New-Item -ItemType Directory -Path $rustDir -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $repoRoot "rust/hu_m3_engine") -Destination $rustDir -Recurse
    Copy-Item -LiteralPath (Join-Path $repoRoot "rust/ofc_stage3_feature_encoder") -Destination $rustDir -Recurse
    $modelEntries = @()
    foreach ($relative in $RequiredModels) {
        $source = Join-Path $repoRoot $relative
        if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { throw "Required population model is missing: $relative" }
        $destination = Join-Path $packageDir $relative
        New-Item -ItemType Directory -Path (Split-Path $destination -Parent) -Force | Out-Null
        Copy-Item -LiteralPath $source -Destination $destination
        $modelEntries += [ordered]@{ path = $relative; bytes = [long](Get-Item $source).Length; sha256 = Get-Sha256 $source }
    }
    foreach ($relative in $RequiredNativeBinaries) {
        $source = Join-Path $repoRoot $relative
        if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { throw "Required Linux native binary is missing: $relative" }
        Assert-LinuxX8664Elf $source
        $destination = Join-Path $packageDir $relative
        New-Item -ItemType Directory -Path (Split-Path $destination -Parent) -Force | Out-Null
        Copy-Item -LiteralPath $source -Destination $destination
    }
    $artifactDir = Join-Path $packageDir "artifacts"
    New-Item -ItemType Directory -Path $artifactDir -Force | Out-Null
    Copy-Item -LiteralPath $resolved.model -Destination (Join-Path $artifactDir "m43_model.pkl")
    foreach ($name in $resolved.Keys | Where-Object { $_ -ne 'model' }) {
        Copy-Item -LiteralPath $resolved[$name] -Destination (Join-Path $artifactDir ("$name.json"))
    }
    $frozenPlanPath = Join-Path $artifactDir "plan.json"
    Copy-Item -LiteralPath $frozenPlanPath -Destination (Join-Path $packageDir "configs/hu_joint_policy_m43_population.json") -Force
    $copiedArtifactHashes = [ordered]@{
        model = Get-Sha256 (Join-Path $artifactDir "m43_model.pkl")
        training_manifest = Get-Sha256 (Join-Path $artifactDir "training_manifest.json")
        data_contract = Get-Sha256 (Join-Path $artifactDir "data_contract.json")
        freeze_manifest = Get-Sha256 (Join-Path $artifactDir "freeze_manifest.json")
        locked_receipt = Get-Sha256 (Join-Path $artifactDir "locked_receipt.json")
        consumption_marker = Get-Sha256 (Join-Path $artifactDir "consumption_marker.json")
        plan = Get-Sha256 (Join-Path $artifactDir "plan.json")
    }
    if ($copiedArtifactHashes.model -ne $preflight.model_sha256 -or
        $copiedArtifactHashes.training_manifest -ne $preflight.training_manifest_sha256 -or
        $copiedArtifactHashes.data_contract -ne $preflight.data_contract_file_sha256 -or
        $copiedArtifactHashes.freeze_manifest -ne $preflight.freeze_manifest_file_sha256 -or
        $copiedArtifactHashes.locked_receipt -ne $preflight.locked_receipt_file_sha256 -or
        $copiedArtifactHashes.consumption_marker -ne $preflight.consumption_marker_sha256 -or
        $copiedArtifactHashes.plan -ne $preflight.population_plan_file_sha256) {
        throw "M4.3 lifecycle artifacts changed while the immutable package was assembled"
    }

    $shardSpecs = @()
    for ($index = 0; $index -lt [int]$plan.shards; $index++) {
        $offset = $index * [int]$plan.paired_seeds_per_shard
        $shardSpecs += [ordered]@{
            shard = $index
            offset = $offset
            seed = [long]$plan.seed + [long]$offset * [long]$plan.seed_stride
            seed_stride = [long]$plan.seed_stride
            paired_seeds = [int]$plan.paired_seeds_per_shard
            output_prefix = ("shard-{0:D4}" -f $index)
        }
    }
    Write-Utf8NoBom $shardManifestPath ((@($shardSpecs | ForEach-Object { $_ | ConvertTo-Json -Compress }) -join "`n") + "`n")
    Copy-Item -LiteralPath $shardManifestPath -Destination (Join-Path $packageDir "population_shards.jsonl")
    Write-Utf8NoBom (Join-Path $packageDir "population_source_models.json") (($modelEntries | ConvertTo-Json -Depth 5) + "`n")
    New-ZipWithForwardSlashes $packageDir $sourceArchivePath

    $startup = @'
#!/usr/bin/env bash
set -Eeuo pipefail
LOG=/var/log/hu_m4_population_startup.log
exec > >(tee -a "$LOG") 2>&1
meta(){ curl -fsS -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }
RUN_NAME="$(meta RUN_NAME)"; BUCKET="$(meta BUCKET)"; SHARD="$(meta SHARD)"
SOURCE_URI="$(meta SOURCE_URI)"; SOURCE_SHA="$(meta SOURCE_SHA)"; MANIFEST_SHA="$(meta MANIFEST_SHA)"
SYNC_SECONDS="$(meta SYNC_SECONDS)"; SELF_DELETE="$(meta SELF_DELETE)"
INSTANCE="$(curl -fsS -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/name)"
ZONE_URL="$(curl -fsS -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/zone)"; ZONE="${ZONE_URL##*/}"
PREFIX="gs://${BUCKET}/runs/${RUN_NAME}"; OUT=/work/out; WORK=/work/repo; mkdir -p "$OUT"
STATUS="$OUT/status.json"; HEARTBEAT_PID=""
write_status(){ python3 - "$STATUS" "$RUN_NAME" "$SHARD" "$1" "$2" "$INSTANCE" "$ZONE" "$MANIFEST_SHA" <<'PY'
import datetime,json,sys
p,run,shard,state,code,instance,zone,manifest=sys.argv[1:]
json.dump({'schema':'hu_m4_population_spot_status_v1','run_name':run,'shard':int(shard),'state':state,'exit_code':int(code),'instance':instance,'zone':zone,'manifest_sha256':manifest,'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat()},open(p,'w'),sort_keys=True)
PY
gcloud storage cp "$STATUS" "$PREFIX/status/shard-${SHARD}.json" >/dev/null || true; }
cleanup(){ code=$?; [[ -z "$HEARTBEAT_PID" ]] || kill "$HEARTBEAT_PID" >/dev/null 2>&1 || true; if [[ $code -ne 0 ]]; then write_status failed "$code"; gcloud storage cp "$LOG" "$PREFIX/results/shard-$(printf '%04d' "$SHARD")/startup.log" >/dev/null || true; fi; if [[ "$SELF_DELETE" == 1 ]]; then gcloud compute instances delete "$INSTANCE" --zone "$ZONE" --quiet >/dev/null 2>&1 || true; fi; exit "$code"; }
trap cleanup EXIT
write_status booting 0
if gcloud storage ls "$PREFIX/results/shard-$(printf '%04d' "$SHARD")/DONE" >/dev/null 2>&1; then write_status complete 0; exit 0; fi
export DEBIAN_FRONTEND=noninteractive
apt-get update -y; apt-get install -y unzip curl ca-certificates python3 python3-venv python3-pip
if ! command -v gcloud >/dev/null 2>&1; then
  apt-get install -y apt-transport-https gnupg
  curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
  echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" >/etc/apt/sources.list.d/google-cloud-sdk.list
  apt-get update -y; apt-get install -y google-cloud-cli
fi
gcloud storage cp "$SOURCE_URI" /tmp/source.zip >/dev/null
echo "${SOURCE_SHA}  /tmp/source.zip" | sha256sum -c -
rm -rf "$WORK"; mkdir -p "$WORK"; unzip -q /tmp/source.zip -d "$WORK"; cd "$WORK"
SPEC="$(sed -n "$((SHARD+1))p" population_shards.jsonl)"; [[ -n "$SPEC" ]]
eval "$(python3 - "$SPEC" <<'PY'
import json,shlex,sys
x=json.loads(sys.argv[1])
for k,v in {'SEED':x['seed'],'STRIDE':x['seed_stride'],'COUNT':x['paired_seeds'],'OUTPUT_PREFIX':x['output_prefix']}.items(): print(f'{k}={shlex.quote(str(v))}')
PY
)"
python3 -m venv .venv; source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'numpy==2.2.6' 'scikit-learn==1.8.0'
python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.6.0'
export PYTHONPATH="$WORK/src" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 OFC_HU_M3_BATCH_THREADS=4
python - <<'PY'
from ofc_regular.hu_m3_rust import engine_version, load_native_engine
from ofc_regular.hu_turn3_stage3_feature_rust import rust_direct_available
print('hu_m3_engine='+engine_version(library=load_native_engine()))
assert rust_direct_available(), 'Stage3 native feature encoder unavailable'
PY
MODEL_SHA="$(sha256sum artifacts/m43_model.pkl | cut -d' ' -f1)"
write_status evaluating 0
(while true; do sleep "$SYNC_SECONDS"; write_status evaluating 0; done) & HEARTBEAT_PID=$!
python -m ofc_regular.evaluate_hu_m4_population --model artifacts/m43_model.pkl --paired-seeds "$COUNT" --seed "$SEED" --seed-stride "$STRIDE" --opponents stage19_p0 stage9f_p2 stage7_m5_r10 random_exact_final --records-output "$OUT/records.jsonl" --output "$OUT/evaluation.json" --progress-every 10 > "$OUT/evaluation_stdout.log"
kill "$HEARTBEAT_PID" >/dev/null 2>&1 || true; HEARTBEAT_PID=""
python - "$OUT/evaluation.json" "$OUT/records.jsonl" "$SEED" "$STRIDE" "$COUNT" "$MODEL_SHA" <<'PY'
import json,sys
e=json.load(open(sys.argv[1])); rows=[json.loads(x) for x in open(sys.argv[2]) if x.strip()]
seed,stride,count=int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5]); model=sys.argv[6]
assert e['schema']=='hu_m4_t1_population_evaluation_v1' and e['paired_seat_swap'] is True
assert e['seed']==seed and e['seed_stride']==stride and e['paired_seeds_per_opponent']==count
assert e['opponents']==['stage19_p0','stage9f_p2','stage7_m5_r10','random_exact_final']
assert len(rows)==count*8 and e['trace_hands']==count*16
r=e['runtime_config']; assert r['current_profile_used'] is False and r['promotion_artifact_contract'] is True and r['diagnostic_legacy'] is False
assert r['candidate_model_sha256']==model and r['safety_model_sha256']==model
assert r['action_score_mode']=='baseline_paired_delta_risk_ensemble_v3'
PY
EVAL_SHA="$(sha256sum "$OUT/evaluation.json" | cut -d' ' -f1)"; RECORDS_SHA="$(sha256sum "$OUT/records.jsonl" | cut -d' ' -f1)"
python - "$OUT/DONE" "$RUN_NAME" "$SHARD" "$MANIFEST_SHA" "$SOURCE_SHA" "$MODEL_SHA" "$EVAL_SHA" "$RECORDS_SHA" <<'PY'
import json,sys
p,run,shard,manifest,source,model,evaluation,records=sys.argv[1:]
json.dump({'schema':'hu_m4_population_spot_done_v1','status':'complete','run_name':run,'shard':int(shard),'manifest_sha256':manifest,'source_sha256':source,'model_sha256':model,'evaluation_sha256':evaluation,'records_sha256':records,'current_profile_mutated':False,'no_runtime_activation':True},open(p,'w'),sort_keys=True)
PY
RESULT="$PREFIX/results/$OUTPUT_PREFIX"
gcloud storage cp "$OUT/evaluation.json" "$RESULT/evaluation.json" >/dev/null
gcloud storage cp "$OUT/records.jsonl" "$RESULT/records.jsonl" >/dev/null
gcloud storage cp "$OUT/evaluation_stdout.log" "$RESULT/evaluation_stdout.log" >/dev/null
gcloud storage cp "$LOG" "$RESULT/startup.log" >/dev/null
write_status complete 0
gcloud storage cp "$OUT/DONE" "$RESULT/DONE" --if-generation-match=0 >/dev/null
'@
    Write-Utf8NoBom $startupPath ($startup + "`n")

    $manifest = [ordered]@{
        schema = "hu_m4_population_spot_manifest_v1"
        run_name = $RunName
        project_id = $ProjectId
        bucket = $Bucket
        population_plan = [ordered]@{ uri = "$gcsPrefix/source/population_plan.json"; sha256 = [string]$preflight.population_plan_file_sha256; paired_seeds = [int]$plan.paired_seeds_per_opponent; seed = [long]$plan.seed; seed_stride = [long]$plan.seed_stride; shards = [int]$plan.shards; paired_seeds_per_shard = [int]$plan.paired_seeds_per_shard }
        source = [ordered]@{ uri = $sourceUri; sha256 = Get-Sha256 $sourceArchivePath; bytes = [long](Get-Item $sourceArchivePath).Length }
        startup = [ordered]@{ uri = $startupUri; sha256 = Get-Sha256 $startupPath }
        shards = [ordered]@{ uri = $shardsUri; sha256 = Get-Sha256 $shardManifestPath; count = [int]$plan.shards }
        runtime = [ordered]@{ model_sha256 = [string]$preflight.model_sha256; frozen_threshold = [double]$freeze.frozen_threshold; training_manifest_sha256 = [string]$preflight.training_manifest_sha256; data_contract_sha256 = [string]$contract.contract_sha256; freeze_manifest_sha256 = [string]$preflight.freeze_manifest_file_sha256; locked_receipt_sha256 = [string]$preflight.locked_receipt_file_sha256; consumption_marker_sha256 = [string]$preflight.consumption_marker_sha256; current_profile_used = $false }
        launch_preflight = [ordered]@{ schema = [string]$preflight.schema; status = [string]$preflight.status; model_sha256 = [string]$preflight.model_sha256; training_manifest_sha256 = [string]$preflight.training_manifest_sha256; data_contract_file_sha256 = [string]$preflight.data_contract_file_sha256; population_plan_file_sha256 = [string]$preflight.population_plan_file_sha256; freeze_manifest_canonical_sha256 = [string]$preflight.freeze_manifest_canonical_sha256; locked_receipt_canonical_sha256 = [string]$preflight.locked_receipt_canonical_sha256; consumption_marker_canonical_sha256 = [string]$preflight.consumption_marker_canonical_sha256; freeze_manifest_file_sha256 = [string]$preflight.freeze_manifest_file_sha256; locked_receipt_file_sha256 = [string]$preflight.locked_receipt_file_sha256; consumption_marker_sha256 = [string]$preflight.consumption_marker_sha256; teacher_hand_seeds_checked = [int]$preflight.teacher_hand_seeds_checked; population_hand_seeds_checked = [int]$preflight.population_hand_seeds_checked; teacher_overlap_count = [int]$preflight.teacher_overlap_count; prior_population_overlap_count = [int]$preflight.prior_population_overlap_count }
        compute = [ordered]@{ machine_type = $MachineType; fallback_machine_types = $FallbackMachineTypes; zones = $Zones; boot_disk_gb = $BootDiskGb; provisioning_model = "SPOT"; instance_termination_action = "DELETE"; self_delete = (-not [bool]$NoSelfDelete); sync_interval_seconds = $SyncIntervalSeconds }
        checkpoint = [ordered]@{ unit = "completed_shard"; retry = "deterministic_full_shard"; resume_missing_shards_only = $true; done_commit_last = $true }
        no_runtime_activation = $true
        current_profile_mutated = $false
    }
    Write-Utf8NoBom $manifestPath (($manifest | ConvertTo-Json -Depth 12) + "`n")
}

$manifestSha = Get-Sha256 $manifestPath
if ($PackageOnly -or $DryRun) {
    [ordered]@{ run_name = $RunName; mode = $(if ($DryRun) { 'dry_run' } else { 'package_only' }); run_dir = $runDir; manifest_sha256 = $manifestSha; source_sha256 = Get-Sha256 $sourceArchivePath; shards = [int]$manifest.shards.count; no_runtime_activation = $true } | ConvertTo-Json -Depth 6
    exit 0
}

if (-not $ResumeExisting) {
    foreach ($pair in @(
        @($sourceArchivePath, $sourceUri), @($startupPath, $startupUri), @($shardManifestPath, $shardsUri), @($frozenPlanPath, "$gcsPrefix/source/population_plan.json")
    )) {
        if (Test-GcsObject $pair[1]) { throw "Immutable population object already exists: $($pair[1])" }
        Invoke-Gcloud @("storage", "cp", $pair[0], $pair[1], "--project", $ProjectId, "--if-generation-match=0") | Out-Null
    }
    if (Test-GcsObject $manifestUri) { throw "Immutable population manifest already exists: $manifestUri" }
    Invoke-Gcloud @("storage", "cp", $manifestPath, $manifestUri, "--project", $ProjectId, "--if-generation-match=0") | Out-Null
}

if (-not $CreateInstances) {
    [ordered]@{ run_name = $RunName; state = "packaged_and_uploaded"; manifest_sha256 = $manifestSha; create_instances = $false; no_runtime_activation = $true } | ConvertTo-Json -Depth 6
    exit 0
}

$selected = if ($StartShards.Count) { @($StartShards | ForEach-Object { [int]$_ }) } else { @(0..([int]$manifest.shards.count - 1)) }
foreach ($shard in $selected) {
    if ($shard -lt 0 -or $shard -ge [int]$manifest.shards.count) { throw "Shard index outside frozen plan: $shard" }
    $outputPrefix = "shard-{0:D4}" -f $shard
    if (Test-GcsObject "$gcsPrefix/results/$outputPrefix/DONE") { continue }
    $vmName = "$vmPrefix-s{0:D3}" -f $shard
    $existing = Invoke-Gcloud @("compute", "instances", "list", "--project", $ProjectId, "--filter", "name=$vmName", "--format=value(name)")
    if (@($existing | Where-Object { $_ }).Count) {
        if ($SkipExistingInstances) { continue }
        throw "Population worker already exists: $vmName"
    }
    $created = $false; $errors = @(); $attempt = 0
    foreach ($machine in @($MachineType) + @($FallbackMachineTypes)) {
        $zone = $Zones[$attempt % $Zones.Count]; $attempt++
        $args = @("compute", "instances", "create", $vmName, "--project", $ProjectId, "--zone", $zone, "--machine-type", $machine, "--provisioning-model", "SPOT", "--instance-termination-action", "DELETE", "--boot-disk-size", "${BootDiskGb}GB", "--image-family", "debian-12", "--image-project", "debian-cloud", "--scopes", "cloud-platform", "--metadata", "RUN_NAME=$RunName,BUCKET=$Bucket,SHARD=$shard,SOURCE_URI=$($manifest.source.uri),SOURCE_SHA=$($manifest.source.sha256),MANIFEST_SHA=$manifestSha,SYNC_SECONDS=$SyncIntervalSeconds,SELF_DELETE=$(if ($NoSelfDelete) { 0 } else { 1 })", "--metadata-from-file", "startup-script=$startupPath")
        try { Invoke-Gcloud $args | Out-Null; $created = $true; break } catch { $errors += $_.Exception.Message }
    }
    if (-not $created) { throw "Could not create population worker $vmName`n$($errors -join [Environment]::NewLine)" }
}

[ordered]@{ run_name = $RunName; state = "workers_started"; shards = $selected; manifest_sha256 = $manifestSha; provisioning_model = "SPOT"; no_runtime_activation = $true; current_profile_mutated = $false } | ConvertTo-Json -Depth 8
