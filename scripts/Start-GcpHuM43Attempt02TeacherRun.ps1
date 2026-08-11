param(
    [string]$ProjectId = "ofc-solver-485418",
    [string]$Bucket = "pokerhu-ofc-solver-485418-training",
    [string]$RunName = ("regular-hu-m43-attempt02-c2e64-pilot300-" + (Get-Date -Format "yyyyMMdd-HHmmss")),
    [string]$PlanPath = "configs/hu_joint_policy_m43_attempt02.json",
    [string]$Attempt01DataContractPath = "outputs/hu_joint_policy/m43_spot/regular-hu-m43-c2e16-pilot200-20260713-1810/m43_t1_second_attempt01/data_contract.json",
    [string]$Attempt01TrainingManifestPath = "outputs/hu_joint_policy/m43_fold_model_runs/regular-hu-m43-fold-model-20260713-2052/training_manifest.json",
    [string]$Attempt01TrainPath = "outputs/hu_joint_policy/m43_spot/regular-hu-m43-c2e16-pilot200-20260713-1810/train.jsonl",
    [string]$Attempt01CalibrationPath = "outputs/hu_joint_policy/m43_spot/regular-hu-m43-c2e16-pilot200-20260713-1810/calibration.jsonl",
    [string]$InheritedLockedPath = "outputs/hu_joint_policy/m43_spot/regular-hu-m43-c2e16-pilot200-20260713-1810/locked_holdout.jsonl",
    [string]$PreflightReceiptPath,
    [int]$NativeBatchThreads = 4,
    [int]$SyncIntervalSeconds = 60,
    [string]$MachineType = "c4-standard-4",
    [string]$BootDiskType = "hyperdisk-balanced",
    [string[]]$FallbackMachineTypes = @("n2-standard-4", "e2-standard-4"),
    [string]$FallbackBootDiskType = "pd-balanced",
    [string[]]$Zones = @("asia-northeast1-b", "asia-northeast1-c", "us-central1-a"),
    [int]$BootDiskGb = 50,
    [string[]]$InitialShards = @("0"),
    [string[]]$StartShards = @(),
    [switch]$CreateInstances,
    [switch]$SkipExistingInstances,
    [switch]$NoSelfDelete,
    [switch]$PackageOnly,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$RootProfiles = @(
    "stage19_p0", "stage9f_p2", "stage7_m5_r10",
    "stage3_baseline", "random_exact_final"
)
$RootProfileWeights = @(1.0, 1.0, 1.0, 1.0, 1.0)
$BaselineProfile = "stage18_p1"
$T2Profile = "stage9f_p2"
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
$PinnedNativeArtifacts = @{
    "target/release/libofc_stage3_feature_encoder.so" = [pscustomobject]@{
        bytes = 500712; sha256 = "9e58797bd234f9858fdfeee69fee03f6d0d20b6e3082de5ab54e9a6356ad0ea7"
    }
    "target/release/libofc_hu_m3_engine.so" = [pscustomobject]@{
        bytes = 1124776; sha256 = "70523749b757a92b8547a5753460c0d9c3435b6872ca76c231a8c38ecc1bbc36"
    }
}

function Write-Utf8NoBom {
    param([string]$Path, [string]$Text)
    [System.IO.File]::WriteAllText($Path, $Text, [System.Text.UTF8Encoding]::new($false))
}

function Get-Sha256 {
    param([string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Assert-CanonicalSelfHash {
    param([string]$Path, [string]$Field)
    $validator = @'
import hashlib,json,sys
path,field=sys.argv[1:]
obj=json.load(open(path,encoding="utf-8")); claimed=obj.pop(field,None)
actual=hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(",",":")).encode()).hexdigest()
assert claimed == actual, (claimed,actual)
'@
    $saved = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $output = @($validator | & python - $Path $Field 2>&1)
        $code = $LASTEXITCODE
    }
    finally { $ErrorActionPreference = $saved }
    if ($code -ne 0) { throw "Canonical self-hash validation failed: $($output -join "`n")" }
}

function Assert-LinuxX8664Elf {
    param([string]$Path)
    $stream = [System.IO.File]::OpenRead($Path)
    try {
        $header = New-Object byte[] 20
        if ($stream.Read($header, 0, $header.Length) -ne $header.Length -or
            $header[0] -ne 0x7f -or $header[1] -ne 0x45 -or
            $header[2] -ne 0x4c -or $header[3] -ne 0x46 -or
            $header[4] -ne 2 -or $header[5] -ne 1 -or
            $header[16] -ne 0x03 -or $header[17] -ne 0x00 -or
            $header[18] -ne 0x3e -or $header[19] -ne 0x00) {
            throw "Pinned native binary is not a Linux x86_64 ELF shared library: $Path"
        }
    }
    finally { $stream.Dispose() }
}

function Convert-ToVmPrefix {
    param([string]$Name)
    $value = ($Name.ToLowerInvariant() -replace '[^a-z0-9-]', '-').Trim('-')
    if (-not $value) { throw "RunName does not produce a valid VM name" }
    if ($value.Length -gt 54) { $value = $value.Substring(0, 54).Trim('-') }
    return $value
}

function Convert-ShardSelection {
    param([string[]]$Values, [string]$ParameterName)
    $indices = @()
    foreach ($value in $Values) {
        foreach ($part in ($value -split ',')) {
            if (-not $part.Trim()) { continue }
            $parsed = 0
            if (-not [int]::TryParse($part.Trim(), [ref]$parsed)) {
                throw "$ParameterName contains a non-integer shard: $part"
            }
            $indices += $parsed
        }
    }
    return @($indices | Sort-Object -Unique)
}

if ($RunName -notmatch '^[a-zA-Z0-9][a-zA-Z0-9._-]*$') {
    throw "RunName may only contain letters, numbers, dot, underscore, and dash"
}
if ($NativeBatchThreads -lt 1 -or $NativeBatchThreads -gt 64) { throw "NativeBatchThreads must be in [1,64]" }
if ($SyncIntervalSeconds -lt 10) { throw "SyncIntervalSeconds must be at least 10" }
if ($BootDiskGb -lt 20) { throw "BootDiskGb must be at least 20" }
if (-not $Zones -or $Zones.Count -eq 0) { throw "At least one zone is required" }
if ($PackageOnly -and $CreateInstances) { throw "PackageOnly and CreateInstances are mutually exclusive" }
if ($DryRun -and $CreateInstances) { throw "DryRun never creates instances" }
if ($StartShards.Count -gt 0 -and $InitialShards.Count -gt 0 -and
    -not ($InitialShards.Count -eq 1 -and [string]$InitialShards[0] -eq "0")) {
    throw "InitialShards and StartShards are mutually exclusive"
}
if ($RequiredModels.Count -ne 11 -or $RequiredNativeBinaries.Count -ne 2) {
    throw "Attempt02 requires exactly 11 models and two native binaries"
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runDir = Join-Path $repoRoot "outputs/gcp_runs/$RunName"
$explicitResume = $StartShards.Count -gt 0
$selectedShardIndices = if ($explicitResume) {
    @(Convert-ShardSelection -Values $StartShards -ParameterName "StartShards")
}
else { @(Convert-ShardSelection -Values $InitialShards -ParameterName "InitialShards") }
if ($selectedShardIndices.Count -eq 0 -and -not $PackageOnly) { throw "At least one shard must be selected" }

$packageDir = Join-Path $runDir "package_src"
$packagePath = Join-Path $runDir "ofc_regular_hu_m43_attempt02_teacher_source.zip"
$startupPath = Join-Path $runDir "startup_hu_m43_attempt02_teacher.sh"
$manifestPath = Join-Path $runDir "manifest.json"
$shardManifestPath = Join-Path $runDir "shards_manifest.jsonl"
$modelManifestPath = Join-Path $runDir "source_model_manifest.json"
$nativeManifestPath = Join-Path $runDir "source_native_manifest.json"
$prefix = "gs://$Bucket/runs/$RunName"
$sourceUri = "$prefix/source/ofc_regular_hu_m43_attempt02_teacher_source.zip"
$startupUri = "$prefix/source/startup_hu_m43_attempt02_teacher.sh"
$manifestUri = "$prefix/manifest.json"
$shardManifestUri = "$prefix/source/shards_manifest.jsonl"
$modelManifestUri = "$prefix/source/source_model_manifest.json"
$nativeManifestUri = "$prefix/source/source_native_manifest.json"

$cloudSdkGcloud = Join-Path $env:LOCALAPPDATA "Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
if (-not $DryRun -and -not $PackageOnly) {
    if (Test-Path -LiteralPath $cloudSdkGcloud) { Set-Alias -Name gcloud -Value $cloudSdkGcloud -Scope Script }
    elseif (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) { throw "gcloud not found" }
}

if ($explicitResume) {
    if ($DryRun -or $PackageOnly) { throw "StartShards is for an existing uploaded run only" }
    New-Item -ItemType Directory -Force -Path $runDir | Out-Null
    foreach ($pair in @(
        @($manifestUri, $manifestPath), @($sourceUri, $packagePath),
        @($startupUri, $startupPath), @($shardManifestUri, $shardManifestPath),
        @($modelManifestUri, $modelManifestPath), @($nativeManifestUri, $nativeManifestPath)
    )) {
        & gcloud storage cp $pair[0] $pair[1] --project $ProjectId | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Unable to download frozen Attempt02 artifact: $($pair[0])" }
    }
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    if ($manifest.schema -ne "hu_m43_attempt02_teacher_spot_manifest_v1" -or
        [string]$manifest.run_name -ne $RunName -or
        [string]$manifest.project_id -ne $ProjectId -or [string]$manifest.bucket -ne $Bucket) {
        throw "Frozen Attempt02 run identity/schema mismatch"
    }
    $manifestText = $manifest | ConvertTo-Json -Depth 20 -Compress
    if ($manifestText -match '(?i)locked|attempt01|inherited[-_ ]holdout') {
        throw "Frozen Attempt02 cloud manifest contains sealed/local-only material"
    }
    $sourceSha256 = Get-Sha256 $packagePath
    $startupSha256 = Get-Sha256 $startupPath
    $manifestSha256 = Get-Sha256 $manifestPath
    $shardManifestSha256 = Get-Sha256 $shardManifestPath
    $modelManifestSha256 = Get-Sha256 $modelManifestPath
    $nativeManifestSha256 = Get-Sha256 $nativeManifestPath
    if ($sourceSha256 -ne [string]$manifest.source_sha256 -or
        $startupSha256 -ne [string]$manifest.startup_sha256 -or
        $shardManifestSha256 -ne [string]$manifest.shards_manifest_sha256 -or
        $modelManifestSha256 -ne [string]$manifest.model_manifest_sha256 -or
        $nativeManifestSha256 -ne [string]$manifest.native_manifest_sha256) {
        throw "Frozen Attempt02 source/startup/manifest hash verification failed"
    }
    $shardSpecs = @()
    foreach ($line in [IO.File]::ReadLines((Resolve-Path -LiteralPath $shardManifestPath))) {
        if ($line.Trim()) { $shardSpecs += ($line | ConvertFrom-Json) }
    }
    foreach ($index in $selectedShardIndices) {
        if ($index -lt 0 -or $index -ge $shardSpecs.Count) { throw "StartShards outside frozen range: $index" }
    }
    $MachineType = [string]$manifest.machine_type
    $BootDiskType = [string]$manifest.boot_disk_type
    $FallbackMachineTypes = @($manifest.fallback_machine_types)
    $FallbackBootDiskType = [string]$manifest.fallback_boot_disk_type
    $Zones = @($manifest.zones)
    $BootDiskGb = [int]$manifest.boot_disk_gb
    $SyncIntervalSeconds = [int]$manifest.sync_interval_seconds
    $NativeBatchThreads = [int]$manifest.native_batch_threads
    $NoSelfDelete = -not [bool]$manifest.self_delete
}
else {
    $resolvedPlan = (Resolve-Path -LiteralPath (Join-Path $repoRoot $PlanPath) -ErrorAction SilentlyContinue)
    if (-not $resolvedPlan) { $resolvedPlan = Resolve-Path -LiteralPath $PlanPath }
    $PlanPath = $resolvedPlan.Path
    foreach ($name in @(
        "Attempt01DataContractPath", "Attempt01TrainingManifestPath",
        "Attempt01TrainPath", "Attempt01CalibrationPath", "InheritedLockedPath"
    )) {
        $value = Get-Variable -Name $name -ValueOnly
        $candidate = Join-Path $repoRoot $value
        if (Test-Path -LiteralPath $candidate) { Set-Variable -Name $name -Value (Resolve-Path -LiteralPath $candidate).Path }
        else { Set-Variable -Name $name -Value (Resolve-Path -LiteralPath $value).Path }
    }
    $temporaryPreflight = $false
    if (-not $PreflightReceiptPath) {
        if ($DryRun) {
            $PreflightReceiptPath = Join-Path ([IO.Path]::GetTempPath()) ("hu-m43-a02-preflight-" + [guid]::NewGuid().ToString("N") + ".json")
            $temporaryPreflight = $true
        }
        else { $PreflightReceiptPath = Join-Path $runDir "m43_attempt02_local_preflight_receipt.json" }
    }
    New-Item -ItemType Directory -Force -Path (Split-Path $PreflightReceiptPath -Parent) | Out-Null
    $preflightArgs = @(
        "-B", "-m", "ofc_regular.hu_m43_attempt02_contract", "preflight",
        "--plan", $PlanPath, "--repo-root", $repoRoot,
        "--attempt01-data-contract", $Attempt01DataContractPath,
        "--attempt01-training-manifest", $Attempt01TrainingManifestPath,
        "--attempt01-train", $Attempt01TrainPath,
        "--attempt01-calibration", $Attempt01CalibrationPath,
        "--inherited-locked", $InheritedLockedPath,
        "--output", $PreflightReceiptPath
    )
    $previousPythonPath = $env:PYTHONPATH
    $env:PYTHONPATH = Join-Path $repoRoot "src"
    try {
        & python @preflightArgs | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Attempt02 local preflight failed with exit code $LASTEXITCODE" }
    }
    finally { $env:PYTHONPATH = $previousPythonPath }
    $preflight = Get-Content -LiteralPath $PreflightReceiptPath -Raw | ConvertFrom-Json
    if ($preflight.schema -ne "hu_m43_attempt02_preflight_receipt_v1" -or
        $preflight.status -ne "pass_frozen_before_fresh_generation") {
        throw "Attempt02 preflight receipt schema/status mismatch"
    }
    Assert-CanonicalSelfHash -Path $PreflightReceiptPath -Field "receipt_sha256"
    $preflightFileSha256 = Get-Sha256 $PreflightReceiptPath

    New-Item -ItemType Directory -Force -Path $runDir | Out-Null
    $scheduleBuilder = @'
import collections,hashlib,json,pathlib,sys
plan_path,preflight_path,output_path=sys.argv[1:]
p=json.load(open(plan_path,encoding="utf-8"))
r=json.load(open(preflight_path,encoding="utf-8"))
assert p["schema"]=="hu_m43_attempt02_plan_v1" and p["status"]=="frozen_pre_generation"
assert r["schema"]=="hu_m43_attempt02_preflight_receipt_v1" and r["status"]=="pass_frozen_before_fresh_generation"
assert p["budget"]=={**p["budget"],"fresh_roots":300,"roots_per_shard":10,"fresh_shards":30}
assert p["teacher_search"]["candidate_samples"]==2 and p["teacher_search"]["evaluation_samples"]==64
assert p["teacher_search"]["candidate_evaluation_rng_disjoint"] is True
union=p["freshness_exclusions"]["expected_union"]
assert union["source_records"]==union["unique_identities"]==union["unique_hand_seeds"]==union["unique_observation_fingerprints"]==412
profiles=[r["profile"] for r in p["root_population"]]
assert profiles==["stage19_p0","stage9f_p2","stage7_m5_r10","stage3_baseline","random_exact_final"]
assert "current" not in profiles and p["fixed_baseline_profile"]=="stage18_p1"
assert p["fixed_continuation"]["t2_profile"]=="stage9f_p2"
roots_per_shard=p["budget"]["roots_per_shard"]
expected_generation={
 "fresh_roots":p["budget"]["fresh_roots"],"roots_per_shard":roots_per_shard,"fresh_shards":p["budget"]["fresh_shards"],
 "splits":{split:{"roots":p["fresh_splits"][split]["roots"],"shards":p["fresh_splits"][split]["roots"]//roots_per_shard,**{key:p["fresh_splits"][split][key] for key in ("seed_start","seed_stride","candidate_seed_start","evaluation_seed_start","child_policy_seed_start")},"hand_seed_stride_scope":"root","phase_seed_stride_scope":p["fresh_splits"][split]["phase_seed_stride_scope"]} for split in ("train","calibration")},
 "teacher_search":{key:p["teacher_search"][key] for key in ("candidate_samples","evaluation_samples","common_random_futures","candidate_evaluation_rng_disjoint","batch_child_selectors","native_batch_threads")},
 "root_population":[{"profile":row["profile"],"train_roots":row["roots"]["train"],"calibration_roots":row["roots"]["calibration"]} for row in p["root_population"]],
 "fixed_baseline_profile":p["fixed_baseline_profile"],"fixed_t2_profile":p["fixed_continuation"]["t2_profile"],
}
assert r["fresh_generation"]==expected_generation
fg=r["fresh_generation"]
specs=[]; global_index=0
for split in ("train","calibration"):
    cfg=fg["splits"][split]; roots=cfg["roots"]
    assert roots in (200,100) and roots%roots_per_shard==0
    for split_shard in range(roots//roots_per_shard):
        offset=split_shard*roots_per_shard
        spec={
          "schema":"hu_m43_attempt02_teacher_shard_v1","shard":global_index,
          "split":split,"split_shard":split_shard,"root_offset":offset,
          "roots":roots_per_shard,"seed_start":cfg["seed_start"]+offset*cfg["seed_stride"],
          "seed_stride":cfg["seed_stride"],
          "candidate_seed":cfg["candidate_seed_start"]+split_shard*cfg["seed_stride"],
          "evaluation_seed":cfg["evaluation_seed_start"]+split_shard*cfg["seed_stride"],
          "child_policy_seed":cfg["child_policy_seed_start"]+split_shard*cfg["seed_stride"],
          "seed_derivation":"preflight_projection_hand_root_offset_rng_split_shard_stride_v1",
          "candidate_samples":fg["teacher_search"]["candidate_samples"],
          "evaluation_samples":fg["teacher_search"]["evaluation_samples"],
          "output_prefix":f"{split}_shard_{split_shard:03d}_roots{roots_per_shard:02d}_seed{cfg['seed_start']+offset*cfg['seed_stride']}",
          "root_profiles":profiles,"root_profile_weights":[1.0]*len(profiles),
          "profile_quota_per_shard":{profile:roots_per_shard//len(profiles) for profile in profiles},
          "baseline_profile":p["fixed_baseline_profile"],"t2_profile":p["fixed_continuation"]["t2_profile"],
        }
        assert len({spec["candidate_seed"],spec["evaluation_seed"],spec["child_policy_seed"]})==3
        specs.append(spec); global_index+=1
assert len(specs)==30 and sum(s["roots"] for s in specs)==300
hand_seeds=[s["seed_start"]+i*s["seed_stride"] for s in specs for i in range(s["roots"])]
assert len(hand_seeds)==len(set(hand_seeds))==300
rng_seeds=[s[key] for s in specs for key in ("candidate_seed","evaluation_seed","child_policy_seed")]
assert len(rng_seeds)==len(set(rng_seeds))==90
for split,total in (("train",40),("calibration",20)):
    quota=collections.Counter()
    for s in specs:
        if s["split"]==split: quota.update(s["profile_quota_per_shard"])
    assert quota==collections.Counter({profile:total for profile in profiles})
text="".join(json.dumps(s,sort_keys=True,separators=(",",":"))+"\n" for s in specs)
pathlib.Path(output_path).write_text(text,encoding="utf-8")
print(json.dumps({"schema":"hu_m43_attempt02_cloud_schedule_summary_v1","total_roots":300,"total_shards":30,"split_roots":{"train":200,"calibration":100},"roots_per_shard":10,"candidate_samples":2,"evaluation_samples":64,"schedule_sha256":hashlib.sha256(text.encode()).hexdigest(),"profiles":profiles,"profile_quotas":{"train":{x:40 for x in profiles},"calibration":{x:20 for x in profiles}}},sort_keys=True))
'@
    $saved = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $summaryOutput = @($scheduleBuilder | & python - $PlanPath $PreflightReceiptPath $shardManifestPath 2>&1)
        $summaryCode = $LASTEXITCODE
    }
    finally { $ErrorActionPreference = $saved }
    if ($summaryCode -ne 0) { throw "Attempt02 plan-driven schedule freeze failed: $($summaryOutput -join "`n")" }
    $scheduleSummary = ($summaryOutput -join "`n") | ConvertFrom-Json
    $shardSpecs = @()
    foreach ($line in [IO.File]::ReadLines((Resolve-Path -LiteralPath $shardManifestPath))) {
        if ($line.Trim()) { $shardSpecs += ($line | ConvertFrom-Json) }
    }
    foreach ($index in $selectedShardIndices) {
        if ($index -lt 0 -or $index -ge $shardSpecs.Count) { throw "InitialShards outside frozen range: $index" }
    }

    $modelPlan = @()
    foreach ($relative in $RequiredModels) {
        $path = Join-Path $repoRoot $relative
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Required Attempt02 model is missing: $relative" }
        $modelPlan += [pscustomobject]@{ path = $relative; bytes = [long](Get-Item -LiteralPath $path).Length }
    }
    $nativePlan = @()
    foreach ($relative in $RequiredNativeBinaries) {
        $path = Join-Path $repoRoot $relative
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Required native binary is missing: $relative" }
        Assert-LinuxX8664Elf -Path $path
        $expected = $PinnedNativeArtifacts[$relative]
        if ([long](Get-Item -LiteralPath $path).Length -ne [long]$expected.bytes -or
            (Get-Sha256 $path) -ne [string]$expected.sha256) {
            throw "Native binary does not match pinned Docker artifact: $relative"
        }
        $nativePlan += [pscustomobject]@{ path = $relative; bytes = [long]$expected.bytes; sha256 = [string]$expected.sha256 }
    }

    if ($DryRun) {
        $plan = Get-Content -LiteralPath $PlanPath -Raw | ConvertFrom-Json
        $result = [ordered]@{
            schema = "hu_m43_attempt02_teacher_spot_dry_run_v1"
            status = "pass_no_cloud_mutation"
            run_name = $RunName
            plan_schema = $plan.schema
            preflight_schema = $preflight.schema
            preflight_status = $preflight.status
            preflight_receipt_sha256 = $preflight.receipt_sha256
            exclusion_union_count = [int]$plan.freshness_exclusions.expected_union.unique_identities
            total_roots = $scheduleSummary.total_roots
            split_roots = $scheduleSummary.split_roots
            roots_per_shard = $scheduleSummary.roots_per_shard
            total_shards = $scheduleSummary.total_shards
            selected_shards = $selectedShardIndices
            candidate_samples = $scheduleSummary.candidate_samples
            evaluation_samples = $scheduleSummary.evaluation_samples
            root_profiles = $scheduleSummary.profiles
            profile_quotas = $scheduleSummary.profile_quotas
            schedule_sha256 = $scheduleSummary.schedule_sha256
            cloud_manifest_preflight_binding = "file_sha256_and_canonical_receipt_sha256_only"
            inherited_private_artifact_uploaded = $false
            source_data_uploaded = $false
            spot = $true
            self_delete = -not $NoSelfDelete
            create_instances = $false
            current_profile_mutated = $false
            no_runtime_activation = $true
        }
        $result | ConvertTo-Json -Depth 10
        if ($temporaryPreflight) { Remove-Item -LiteralPath $PreflightReceiptPath -Force -ErrorAction SilentlyContinue }
        exit 0
    }

    if (Test-Path -LiteralPath $packageDir) { Remove-Item -LiteralPath $packageDir -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $packageDir | Out-Null
    $closureBuilder = @'
import ast,json,pathlib,shutil,sys
repo=pathlib.Path(sys.argv[1]).resolve(); destination=pathlib.Path(sys.argv[2]).resolve(); source=repo/"src"/"ofc_regular"
queue=["ofc_regular","ofc_regular.generate_hu_m4_t1_data"]; seen=set(); selected=[]
def module_path(name):
    parts=name.split(".");
    if parts[0]!="ofc_regular": return None
    candidate=source.joinpath(*parts[1:]); path=candidate/"__init__.py" if candidate.is_dir() else candidate.with_suffix(".py")
    return path if path.is_file() else None
def add(name):
    if name.startswith("ofc_regular") and name not in seen and module_path(name) is not None: queue.append(name)
while queue:
    name=queue.pop(0)
    if name in seen: continue
    path=module_path(name); seen.add(name); selected.append(path)
    tree=ast.parse(path.read_text(encoding="utf-8"),filename=str(path)); package=name if path.name=="__init__.py" else name.rsplit(".",1)[0]
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):
            for alias in node.names: add(alias.name)
        elif isinstance(node,ast.ImportFrom):
            if node.level:
                base=package.split(".")
                if node.level>1: base=base[:-(node.level-1)]
                target=".".join(base+(([node.module] if node.module else []))); add(target)
                if node.module is None:
                    for alias in node.names: add(target+"."+alias.name)
            elif node.module: add(node.module)
entries=[]
for path in selected:
    relative=path.relative_to(repo/"src"); target=destination/"src"/relative
    target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(path,target); entries.append("src/"+relative.as_posix())
entries=sorted(entries); print(json.dumps({"entries":entries,"entries_sha256":__import__("hashlib").sha256(json.dumps(entries,separators=(",",":")).encode()).hexdigest()},sort_keys=True))
'@
    $closureOutput = @($closureBuilder | & python - $repoRoot $packageDir 2>&1)
    if ($LASTEXITCODE -ne 0) { throw "Unable to build Attempt02 source closure: $($closureOutput -join "`n")" }
    $closureInfo = ($closureOutput -join "`n") | ConvertFrom-Json
    if ("src/ofc_regular/generate_hu_m4_t1_data.py" -notin @($closureInfo.entries)) { throw "Generator omitted from source closure" }

    $modelEntries = @()
    foreach ($relative in $RequiredModels) {
        $source = Join-Path $repoRoot $relative; $destination = Join-Path $packageDir $relative
        New-Item -ItemType Directory -Force -Path (Split-Path $destination -Parent) | Out-Null
        Copy-Item -LiteralPath $source -Destination $destination
        $modelEntries += [pscustomobject][ordered]@{ path = $relative; bytes = [long](Get-Item $source).Length; sha256 = Get-Sha256 $source }
    }
    $modelManifest = [ordered]@{ schema = "hu_m43_attempt02_source_model_manifest_v1"; model_count = 11; models = $modelEntries }
    Write-Utf8NoBom -Path $modelManifestPath -Text (($modelManifest | ConvertTo-Json -Depth 8) + "`n")
    $nativeEntries = @()
    foreach ($relative in $RequiredNativeBinaries) {
        $source = Join-Path $repoRoot $relative; $destination = Join-Path $packageDir $relative
        New-Item -ItemType Directory -Force -Path (Split-Path $destination -Parent) | Out-Null
        Copy-Item -LiteralPath $source -Destination $destination
        $nativeEntries += [pscustomobject][ordered]@{ path = $relative; bytes = [long](Get-Item $source).Length; sha256 = Get-Sha256 $source; platform = "linux-x86_64" }
    }
    $nativeManifest = [ordered]@{ schema = "hu_m43_attempt02_source_native_manifest_v1"; binary_count = 2; binaries = $nativeEntries }
    Write-Utf8NoBom -Path $nativeManifestPath -Text (($nativeManifest | ConvertTo-Json -Depth 8) + "`n")
    Copy-Item -LiteralPath $shardManifestPath -Destination (Join-Path $packageDir "shards_manifest.jsonl")
    Copy-Item -LiteralPath $modelManifestPath -Destination (Join-Path $packageDir "source_model_manifest.json")
    Copy-Item -LiteralPath $nativeManifestPath -Destination (Join-Path $packageDir "source_native_manifest.json")

    $zipBuilder = @'
import pathlib,sys,zipfile
source=pathlib.Path(sys.argv[1]); destination=pathlib.Path(sys.argv[2])
if destination.exists(): destination.unlink()
with zipfile.ZipFile(destination,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
    for path in sorted(p for p in source.rglob("*") if p.is_file()):
        name=path.relative_to(source).as_posix()
        if "\\" in name or name.startswith(("configs/","outputs/","data/")): raise ValueError(name)
        archive.write(path,name)
'@
    $zipOutput = @($zipBuilder | & python - $packageDir $packagePath 2>&1)
    if ($LASTEXITCODE -ne 0) { throw "Attempt02 source archive creation failed: $($zipOutput -join "`n")" }
    $sourceSha256 = Get-Sha256 $packagePath
    $shardManifestSha256 = Get-Sha256 $shardManifestPath
    $modelManifestSha256 = Get-Sha256 $modelManifestPath
    $nativeManifestSha256 = Get-Sha256 $nativeManifestPath

    $startup = @'
#!/usr/bin/env bash
set -euo pipefail
export HOME="${HOME:-/root}"
log(){ echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
meta(){ curl -fsS -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }
imeta(){ curl -fsS -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/$1"; }
RUN_NAME="$(meta RUN_NAME)"; BUCKET="$(meta BUCKET)"; SHARD_INDEX="$(meta SHARD_INDEX)"
SOURCE_URI="$(meta SOURCE_URI)"; SOURCE_SHA256="$(meta SOURCE_SHA256)"; STARTUP_SHA256="$(meta STARTUP_SHA256)"
MANIFEST_SHA256="$(meta MANIFEST_SHA256)"; SHARDS_SHA256="$(meta SHARDS_SHA256)"
MODELS_SHA256="$(meta MODELS_SHA256)"; NATIVE_SHA256="$(meta NATIVE_SHA256)"
SYNC_INTERVAL_SECONDS="$(meta SYNC_INTERVAL_SECONDS)"; NATIVE_BATCH_THREADS="$(meta NATIVE_BATCH_THREADS)"; SELF_DELETE="$(meta SELF_DELETE)"
INSTANCE_NAME="$(imeta name)"; ZONE_PATH="$(imeta zone)"; ZONE="${ZONE_PATH##*/}"
PREFIX="gs://${BUCKET}/runs/${RUN_NAME}"; WORK=/opt/ofc-hu-m43-a02; RESULT=/tmp/ofc-hu-m43-a02-result; SYNC=/tmp/ofc-hu-m43-a02-sync
STARTUP_LOG=/tmp/ofc-hu-m43-a02-startup.log; : > "$STARTUP_LOG"; exec > >(tee -a "$STARTUP_LOG") 2>&1
OUTPUT_PREFIX=""; SPLIT=""; PID=""
completed(){ [[ -s "$RESULT/checkpoint.json" ]] || { echo 0; return; }; python3 -c 'import json,sys;print(int(json.load(open(sys.argv[1])).get("completed_roots",0)))' "$RESULT/checkpoint.json" 2>/dev/null || echo 0; }
status(){ python3 - "$RUN_NAME" "$SHARD_INDEX" "$SPLIT" "$1" "$(completed)" "$OUTPUT_PREFIX" > /tmp/status.json <<'PY'
import json,sys,time
run,shard,split,state,done,prefix=sys.argv[1:]
print(json.dumps({"schema":"hu_m43_attempt02_teacher_status_v1","run_name":run,"shard":int(shard),"split":split,"status":state,"completed_roots":int(done),"output_prefix":prefix,"updated_unix_seconds":time.time()},sort_keys=True))
PY
gcloud storage cp /tmp/status.json "$PREFIX/status/shard_${SHARD_INDEX}.json" >/dev/null || true; }
sync_resume(){ [[ -n "$OUTPUT_PREFIX" && -s "$RESULT/teacher.jsonl.partial" && -s "$RESULT/checkpoint.json" ]] || return 0; rm -rf "$SYNC"; mkdir -p "$SYNC"; cp "$RESULT/teacher.jsonl.partial" "$RESULT/checkpoint.json" "$SYNC/"; [[ -s "$RESULT/heartbeat.json" ]] && cp "$RESULT/heartbeat.json" "$SYNC/" || true; expected="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("partial_sha256",""))' "$SYNC/checkpoint.json")"; actual="$(sha256sum "$SYNC/teacher.jsonl.partial"|awk '{print $1}')"; [[ -n "$expected" && "$expected" == "$actual" ]] || return 0; gcloud storage cp "$SYNC/"* "$PREFIX/resume/${OUTPUT_PREFIX}/" >/dev/null; status running; }
cleanup(){ code=$?; set +e; [[ $code -eq 0 ]] || { sync_resume; gcloud storage cp "$STARTUP_LOG" "$PREFIX/logs/startup_shard_${SHARD_INDEX}.log" >/dev/null 2>&1 || true; status failed; }; if [[ "$SELF_DELETE" == 1 ]]; then gcloud compute instances delete "$INSTANCE_NAME" --zone "$ZONE" --quiet >/dev/null 2>&1 || sudo shutdown -h now; fi; exit $code; }
trap cleanup EXIT
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -y; sudo apt-get install -y unzip curl ca-certificates python3 python3-venv python3-pip
if ! command -v gcloud >/dev/null 2>&1; then
  sudo apt-get install -y apt-transport-https gnupg
  curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
  echo 'deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main' | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list >/dev/null
  sudo apt-get update -y; sudo apt-get install -y google-cloud-cli
fi
sudo rm -rf "$WORK"; sudo mkdir -p "$WORK"; sudo chown "$(id -u):$(id -g)" "$WORK"
gcloud storage cp "$SOURCE_URI" /tmp/source.zip >/dev/null; [[ "$(sha256sum /tmp/source.zip|awk '{print $1}')" == "$SOURCE_SHA256" ]]
unzip -q /tmp/source.zip -d "$WORK"; cd "$WORK"
[[ "$(sha256sum shards_manifest.jsonl|awk '{print $1}')" == "$SHARDS_SHA256" ]]
[[ "$(sha256sum source_model_manifest.json|awk '{print $1}')" == "$MODELS_SHA256" ]]
[[ "$(sha256sum source_native_manifest.json|awk '{print $1}')" == "$NATIVE_SHA256" ]]
python3 -m venv .venv; source .venv/bin/activate; python -m pip install --upgrade pip
python -m pip install 'numpy==2.2.6' 'scikit-learn==1.8.0'; python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.6.0'
export PYTHONPATH="$WORK/src" OMP_NUM_THREADS="$NATIVE_BATCH_THREADS" MKL_NUM_THREADS="$NATIVE_BATCH_THREADS" OPENBLAS_NUM_THREADS="$NATIVE_BATCH_THREADS" OFC_HU_M3_BATCH_THREADS="$NATIVE_BATCH_THREADS"
python - <<'PY'
import hashlib,json,pathlib
for name,schema,count in (("source_model_manifest.json","hu_m43_attempt02_source_model_manifest_v1",11),("source_native_manifest.json","hu_m43_attempt02_source_native_manifest_v1",2)):
 m=json.load(open(name)); assert m["schema"]==schema; rows=m["models"] if "models" in m else m["binaries"]; assert len(rows)==count
 for row in rows:
  p=pathlib.Path(row["path"]); assert p.is_file() and p.stat().st_size==row["bytes"] and hashlib.sha256(p.read_bytes()).hexdigest()==row["sha256"]
PY
python - <<'PY'
from ofc_regular.hu_m3_rust import engine_version, load_native_engine
from ofc_regular.hu_turn3_stage3_feature_rust import rust_direct_available
print("hu_m3_engine=" + engine_version(library=load_native_engine()))
assert rust_direct_available(), "Pinned Stage3 feature encoder did not load"
PY
SPEC="$(sed -n "$((SHARD_INDEX+1))p" shards_manifest.jsonl)"; [[ -n "$SPEC" ]]
field(){ python3 -c 'import json,sys;print(json.loads(sys.argv[1])[sys.argv[2]])' "$SPEC" "$1"; }
[[ "$(field shard)" == "$SHARD_INDEX" ]]; SPLIT="$(field split)"; [[ "$SPLIT" == train || "$SPLIT" == calibration ]]
ROOTS="$(field roots)"; SEED_START="$(field seed_start)"; SEED_STRIDE="$(field seed_stride)"; CANDIDATE_SEED="$(field candidate_seed)"; EVALUATION_SEED="$(field evaluation_seed)"; CHILD_SEED="$(field child_policy_seed)"; CANDIDATE_SAMPLES="$(field candidate_samples)"; EVALUATION_SAMPLES="$(field evaluation_samples)"; OUTPUT_PREFIX="$(field output_prefix)"
RESULT_URI="$PREFIX/results/$OUTPUT_PREFIX"; DONE_URI="$RESULT_URI/DONE.json"; RESUME_URI="$PREFIX/resume/$OUTPUT_PREFIX"
if gcloud storage ls "$DONE_URI" >/dev/null 2>&1; then status complete; exit 0; fi
rm -rf "$RESULT" "$SYNC"; mkdir -p "$RESULT" "$SYNC"
if gcloud storage ls "$RESUME_URI/checkpoint.json" >/dev/null 2>&1; then gcloud storage cp "$RESUME_URI/checkpoint.json" "$RESULT/checkpoint.json" >/dev/null; gcloud storage cp "$RESUME_URI/teacher.jsonl.partial" "$RESULT/teacher.jsonl.partial" >/dev/null; gcloud storage cp "$RESUME_URI/heartbeat.json" "$RESULT/heartbeat.json" >/dev/null 2>&1 || true; fi
status running
profiles=(stage19_p0 stage9f_p2 stage7_m5_r10 stage3_baseline random_exact_final); weights=(1 1 1 1 1)
args=(-B -m ofc_regular.generate_hu_m4_t1_data --output "$RESULT/teacher.jsonl" --checkpoint "$RESULT/checkpoint.json" --heartbeat "$RESULT/heartbeat.json" --roots "$ROOTS" --seed-start "$SEED_START" --seed-stride "$SEED_STRIDE" --candidate-samples "$CANDIDATE_SAMPLES" --evaluation-samples "$EVALUATION_SAMPLES" --candidate-seed "$CANDIDATE_SEED" --evaluation-seed "$EVALUATION_SEED" --child-policy-seed "$CHILD_SEED" --root-profiles "${profiles[@]}" --root-profile-weights "${weights[@]}" --baseline-profile stage18_p1 --t2-profile stage9f_p2 --batch-child-selectors --native-batch-threads "$NATIVE_BATCH_THREADS" --opening-lookahead-samples 0 --split "$SPLIT" --run-id "${RUN_NAME}:shard=${SHARD_INDEX}")
set +e; python "${args[@]}" > "$RESULT/generator_summary.json" 2> "$RESULT/run.log" & PID=$!; set -e
while kill -0 "$PID" >/dev/null 2>&1; do for ((i=0;i<SYNC_INTERVAL_SECONDS;i++)); do kill -0 "$PID" >/dev/null 2>&1 || break; sleep 1; done; kill -0 "$PID" >/dev/null 2>&1 && sync_resume || true; done
set +e; wait "$PID"; code=$?; set -e; PID=""; [[ $code -eq 0 ]] || exit $code
[[ -s "$RESULT/teacher.jsonl" && -s "$RESULT/checkpoint.json" && -s "$RESULT/heartbeat.json" ]]
out_sha="$(sha256sum "$RESULT/teacher.jsonl"|awk '{print $1}')"; cp_sha="$(sha256sum "$RESULT/checkpoint.json"|awk '{print $1}')"; hb_sha="$(sha256sum "$RESULT/heartbeat.json"|awk '{print $1}')"; [[ "$(wc -l < "$RESULT/teacher.jsonl"|tr -d ' ')" == "$ROOTS" ]]
python3 - "$RESULT/DONE.json" "$RUN_NAME" "$SHARD_INDEX" "$SPLIT" "$ROOTS" "$OUTPUT_PREFIX" "$out_sha" "$cp_sha" "$hb_sha" "$SOURCE_SHA256" "$STARTUP_SHA256" "$MANIFEST_SHA256" "$SHARDS_SHA256" "$MODELS_SHA256" "$NATIVE_SHA256" <<'PY'
import json,sys,time
(path,run,shard,split,roots,prefix,out_sha,cp_sha,hb_sha,source_sha,startup_sha,manifest_sha,shards_sha,models_sha,native_sha)=sys.argv[1:]
json.dump({"schema":"hu_m43_attempt02_teacher_done_v1","status":"complete","run_name":run,"shard":int(shard),"split":split,"roots":int(roots),"output_prefix":prefix,"output_sha256":out_sha,"checkpoint_sha256":cp_sha,"heartbeat_sha256":hb_sha,"source_sha256":source_sha,"startup_sha256":startup_sha,"manifest_sha256":manifest_sha,"shards_manifest_sha256":shards_sha,"model_manifest_sha256":models_sha,"native_manifest_sha256":native_sha,"completed_unix_seconds":time.time()},open(path,"w"),sort_keys=True)
PY
for file in teacher.jsonl checkpoint.json heartbeat.json generator_summary.json run.log; do gcloud storage cp "$RESULT/$file" "$RESULT_URI/$file" >/dev/null; done
gcloud storage cp "$RESULT/DONE.json" "$DONE_URI" --if-generation-match=0 >/dev/null
status complete; log "complete shard=$SHARD_INDEX split=$SPLIT roots=$ROOTS"
'@
    Write-Utf8NoBom -Path $startupPath -Text $startup
    $startupSha256 = Get-Sha256 $startupPath
    $manifest = [ordered]@{
        schema = "hu_m43_attempt02_teacher_spot_manifest_v1"
        run_name = $RunName; project_id = $ProjectId; bucket = $Bucket
        milestone = "M4.3-attempt02"; purpose = "fresh_train_calibration_teacher_only"
        cloud_input_boundary = "fresh_generation_schedule_only"
        local_preflight_binding = [ordered]@{ file_sha256 = $preflightFileSha256; receipt_sha256 = [string]$preflight.receipt_sha256 }
        total_roots = 300; split_roots = [ordered]@{ train = 200; calibration = 100 }
        roots_per_shard = 10; total_shards = 30; initial_shards = $selectedShardIndices
        candidate_samples = 2; evaluation_samples = 64
        schedule_sha256 = [string]$scheduleSummary.schedule_sha256
        root_profiles = $RootProfiles; root_profile_weights = $RootProfileWeights
        profile_quotas = [ordered]@{ train = $scheduleSummary.profile_quotas.train; calibration = $scheduleSummary.profile_quotas.calibration }
        baseline_profile = $BaselineProfile; t2_profile = $T2Profile
        native_batch_threads = $NativeBatchThreads; sync_interval_seconds = $SyncIntervalSeconds
        machine_type = $MachineType; boot_disk_type = $BootDiskType
        fallback_machine_types = $FallbackMachineTypes; fallback_boot_disk_type = $FallbackBootDiskType
        zones = $Zones; boot_disk_gb = $BootDiskGb; spot = $true; self_delete = -not $NoSelfDelete
        source_uri = $sourceUri; source_sha256 = $sourceSha256
        startup_uri = $startupUri; startup_sha256 = $startupSha256
        shards_manifest_uri = $shardManifestUri; shards_manifest_sha256 = $shardManifestSha256
        model_manifest_uri = $modelManifestUri; model_manifest_sha256 = $modelManifestSha256
        native_manifest_uri = $nativeManifestUri; native_manifest_sha256 = $nativeManifestSha256
        source_closure_entries_sha256 = [string]$closureInfo.entries_sha256
        current_profile_mutated = $false; no_runtime_activation = $true
        created_at = (Get-Date).ToUniversalTime().ToString("o")
    }
    Write-Utf8NoBom -Path $manifestPath -Text (($manifest | ConvertTo-Json -Depth 12) + "`n")
    $manifestSha256 = Get-Sha256 $manifestPath

    $boundaryAudit = @'
import json,pathlib,sys,zipfile
plan_path,package_path,*artifacts=sys.argv[1:]
p=json.load(open(plan_path,encoding="utf-8")); sensitive=[]
def collect(v):
 if isinstance(v,dict):
  for x in v.values(): collect(x)
 elif isinstance(v,list):
  for x in v: collect(x)
 elif isinstance(v,str) and (len(v)>=32 or "/" in v or "\\" in v): sensitive.append(v)
for key in ("freshness_exclusions","attempt01_provenance","inherited_locked"): collect(p[key])
for path in artifacts:
 text=pathlib.Path(path).read_text(encoding="utf-8",errors="ignore")
 for value in sensitive:
  if value and value.lower() in text.lower(): raise ValueError(f"sealed plan value leaked into cloud artifact {path}")
manifest=json.load(open(artifacts[-1],encoding="utf-8"))
serialized=json.dumps(manifest).lower()
assert "locked" not in serialized and "attempt01" not in serialized
with zipfile.ZipFile(package_path) as z:
 names=z.namelist(); assert not any(n.startswith(("configs/","outputs/","data/")) for n in names)
 assert not any("attempt02_contract" in n or "pilot_contract" in n for n in names)
 for name in names:
  if not name.endswith((".py",".json",".jsonl")): continue
  text=z.read(name).decode("utf-8",errors="ignore")
  for value in sensitive:
   if value and value.lower() in text.lower(): raise ValueError(f"sealed plan value leaked into source archive {name}")
print(json.dumps({"schema":"hu_m43_attempt02_cloud_boundary_audit_v1","status":"pass","sensitive_values_uploaded":0,"source_data_uploaded":False,"preflight_receipt_content_uploaded":False},sort_keys=True))
'@
    $boundaryFiles = @($shardManifestPath, $modelManifestPath, $nativeManifestPath, $startupPath, $manifestPath)
    $boundaryOutput = @($boundaryAudit | & python - $PlanPath $packagePath @boundaryFiles 2>&1)
    if ($LASTEXITCODE -ne 0) { throw "Attempt02 cloud boundary audit failed: $($boundaryOutput -join "`n")" }
    $boundaryResult = ($boundaryOutput -join "`n") | ConvertFrom-Json

    if ($PackageOnly) {
        [pscustomobject]@{ schema = "hu_m43_attempt02_teacher_package_result_v1"; run_name = $RunName; package = $packagePath; manifest = $manifestPath; total_shards = 30; cloud_boundary = $boundaryResult; current_profile_mutated = $false } | ConvertTo-Json -Depth 8
        exit 0
    }

    $saved = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $existing = @(& gcloud storage ls $manifestUri --project $ProjectId 2>&1)
        $existsCode = $LASTEXITCODE
    }
    finally { $ErrorActionPreference = $saved }
    if ($existsCode -eq 0) { throw "Attempt02 run already exists and is immutable; use StartShards for resume" }
    if (($existing -join "`n") -notmatch '(?i)(not found|no urls matched|matched no objects|404)') { throw "Unable to prove Attempt02 run absence" }
    foreach ($pair in @(
        @($packagePath, $sourceUri), @($startupPath, $startupUri), @($shardManifestPath, $shardManifestUri),
        @($modelManifestPath, $modelManifestUri), @($nativeManifestPath, $nativeManifestUri), @($manifestPath, $manifestUri)
    )) {
        & gcloud storage cp $pair[0] $pair[1] --project $ProjectId --if-generation-match=0 | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Immutable Attempt02 upload failed: $($pair[1])" }
    }
}

$vmPrefix = Convert-ToVmPrefix $RunName
$instances = @()
foreach ($shardIndex in $selectedShardIndices) {
    $spec = $shardSpecs[$shardIndex]
    $doneUri = "$prefix/results/$($spec.output_prefix)/DONE.json"
    $saved = $ErrorActionPreference
    try { $ErrorActionPreference = "Continue"; & gcloud storage ls $doneUri --project $ProjectId *> $null; $doneCode = $LASTEXITCODE }
    finally { $ErrorActionPreference = $saved }
    if ($doneCode -eq 0) { continue }
    $vmName = ("{0}-{1:D3}" -f $vmPrefix, $shardIndex)
    $metadata = @(
        "RUN_NAME=$RunName", "BUCKET=$Bucket", "SHARD_INDEX=$shardIndex", "SOURCE_URI=$sourceUri",
        "SOURCE_SHA256=$sourceSha256", "STARTUP_SHA256=$startupSha256", "MANIFEST_SHA256=$manifestSha256",
        "SHARDS_SHA256=$shardManifestSha256", "MODELS_SHA256=$modelManifestSha256", "NATIVE_SHA256=$nativeManifestSha256",
        "SYNC_INTERVAL_SECONDS=$SyncIntervalSeconds", "NATIVE_BATCH_THREADS=$NativeBatchThreads",
        ("SELF_DELETE=" + $(if ($NoSelfDelete) { "0" } else { "1" }))
    ) -join ','
    $record = [ordered]@{ name = $vmName; shard = $shardIndex; created = $false; zone = $null; machine_type = $null }
    if ($CreateInstances) {
        $present = @(& gcloud compute instances list --project $ProjectId --filter ("name='" + $vmName + "'") --format "value(name)" 2>$null)
        if ($present.Count -gt 0) {
            if ($SkipExistingInstances) { $instances += [pscustomobject]$record; continue }
            throw "Worker already exists: $vmName"
        }
        $attempts = @([pscustomobject]@{ machine = $MachineType; disk = $BootDiskType })
        foreach ($fallback in $FallbackMachineTypes) { if ($fallback -and $fallback -ne $MachineType) { $attempts += [pscustomobject]@{ machine = $fallback; disk = $FallbackBootDiskType } } }
        foreach ($attempt in $attempts) {
            foreach ($zone in $Zones) {
                & gcloud compute instances create $vmName --project $ProjectId --zone $zone `
                    --machine-type $attempt.machine --image-family ubuntu-2404-lts-amd64 --image-project ubuntu-os-cloud `
                    --boot-disk-size ("{0}GB" -f $BootDiskGb) --boot-disk-type $attempt.disk --boot-disk-auto-delete `
                    --provisioning-model SPOT --instance-termination-action DELETE --maintenance-policy TERMINATE `
                    --scopes cloud-platform --labels purpose=hu-m43-a02-teacher,milestone=m43-a02 `
                    --metadata $metadata --metadata-from-file ("startup-script={0}" -f $startupPath)
                if ($LASTEXITCODE -eq 0) { $record.created = $true; $record.zone = $zone; $record.machine_type = $attempt.machine; break }
            }
            if ($record.created) { break }
        }
        if (-not $record.created) { throw "Unable to create Attempt02 Spot worker: $vmName" }
    }
    $instances += [pscustomobject]$record
}

[pscustomobject]@{
    schema = "hu_m43_attempt02_teacher_spot_start_v1"
    run_name = $RunName; total_roots = [int]$manifest.total_roots; total_shards = $shardSpecs.Count
    selected_shards = $selectedShardIndices; create_instances = [bool]$CreateInstances
    resume_existing = $explicitResume; spot = $true; termination_action = "DELETE"
    manifest_uri = $manifestUri; manifest_sha256 = $manifestSha256
    instances = $instances; current_profile_mutated = $false; no_runtime_activation = $true
} | ConvertTo-Json -Depth 10
