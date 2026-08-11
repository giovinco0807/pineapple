param(
    [Parameter(Mandatory = $true)][string]$ModelPath,
    [Parameter(Mandatory = $true)][string]$FinalTrainingManifestPath,
    [Parameter(Mandatory = $true)][string]$RuntimeFreezePath,
    [Parameter(Mandatory = $true)][string]$TrainingFreezePath,
    [Parameter(Mandatory = $true)][string]$PrecalibrationReceiptPath,
    [Parameter(Mandatory = $true)][string]$ModelFreezePath,
    [string]$Attempt03PlanPath = "configs/hu_joint_policy_m43_attempt03.json",
    [Parameter(Mandatory = $true)][string]$LockedReceiptPath,
    [Parameter(Mandatory = $true)][string]$ConsumptionMarkerPath,
    [string]$PlanPath = "configs/hu_joint_policy_m43_attempt03_population.json",
    [string]$ProjectId = "ofc-solver-485418",
    [string]$Bucket = "pokerhu-ofc-solver-485418-training",
    [string]$RunName = ("regular-hu-m43-attempt03-population-" + (Get-Date -Format "yyyyMMdd-HHmmss")),
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
$ManifestSchema = "hu_m43_attempt03_population_spot_manifest_v1"
$PreflightSchema = "hu_m43_attempt03_population_launch_preflight_v1"
$DoneSchema = "hu_m43_attempt03_population_spot_done_v1"
$ActionScoreMode = "eligible_stage18_stacked_meta_ranker_v5"
$ModelSchema = "hu_m43_t1_joint_model_v5"
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

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}
function Assert-Sha256Text($Value, [string]$Label) {
    if ($Value -isnot [string] -or $Value -cnotmatch '^[0-9a-f]{64}$') {
        throw "$Label must be a lowercase SHA-256 digest"
    }
}
function Write-Utf8NoBom([string]$Path, [string]$Text) {
    [System.IO.File]::WriteAllText($Path, $Text, [System.Text.UTF8Encoding]::new($false))
}
function Invoke-Gcloud([string[]]$Arguments) {
    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { $output = & gcloud @Arguments 2>&1; $code = $LASTEXITCODE }
    finally { $ErrorActionPreference = $old }
    if ($code -ne 0) {
        throw "gcloud failed ($code): gcloud $($Arguments -join ' ')`n$(@($output) -join [Environment]::NewLine)"
    }
    return @($output)
}
function Test-GcsObject([string]$Uri) {
    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { $output = & gcloud storage objects describe $Uri --format=json 2>&1; $code = $LASTEXITCODE }
    finally { $ErrorActionPreference = $old }
    if ($code -eq 0) { return $true }
    if ((@($output) -join "`n") -match '(?i)not found|does not exist|No URLs matched|404') { return $false }
    throw "Could not inspect immutable GCS object: $Uri`n$(@($output) -join [Environment]::NewLine)"
}
function Assert-LinuxX8664Elf([string]$Path) {
    $stream = [System.IO.File]::OpenRead($Path)
    try {
        $header = New-Object byte[] 20
        if ($stream.Read($header, 0, $header.Length) -ne $header.Length -or
            $header[0] -ne 0x7f -or $header[1] -ne 0x45 -or
            $header[2] -ne 0x4c -or $header[3] -ne 0x46 -or
            $header[4] -ne 2 -or $header[5] -ne 1 -or
            $header[16] -ne 0x03 -or $header[17] -ne 0x00 -or
            $header[18] -ne 0x3e -or $header[19] -ne 0x00) {
            throw "Population native dependency is not Linux x86_64 ELF: $Path"
        }
    }
    finally { $stream.Dispose() }
}
function New-ZipWithForwardSlashes([string]$SourceDir, [string]$DestinationPath) {
    Add-Type -AssemblyName System.IO.Compression
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    if (Test-Path -LiteralPath $DestinationPath) { Remove-Item -LiteralPath $DestinationPath -Force }
    $sourceRoot = (Resolve-Path -LiteralPath $SourceDir).Path.TrimEnd('\', '/')
    $prefixLength = $sourceRoot.Length + 1
    $zip = [System.IO.Compression.ZipFile]::Open($DestinationPath, [System.IO.Compression.ZipArchiveMode]::Create)
    try {
        Get-ChildItem -LiteralPath $sourceRoot -Recurse -File | Sort-Object FullName | ForEach-Object {
            $entry = $_.FullName.Substring($prefixLength) -replace '\\', '/'
            [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
                $zip, $_.FullName, $entry, [System.IO.Compression.CompressionLevel]::Optimal
            ) | Out-Null
        }
    }
    finally { $zip.Dispose() }
}
function Convert-ToVmPrefix([string]$Value) {
    $name = ($Value.ToLowerInvariant() -replace '[^a-z0-9-]', '-').Trim('-')
    if (-not $name) { throw "RunName does not produce a VM prefix" }
    if ($name -notmatch '^[a-z]') { $name = "m43a3p-$name" }
    if ($name.Length -gt 56) { $name = $name.Substring(0, 56).TrimEnd('-') }
    return $name
}

if ($RunName -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$') { throw "RunName contains unsafe characters" }
if ($BootDiskGb -lt 20) { throw "BootDiskGb must be at least 20" }
if ($SyncIntervalSeconds -lt 15) { throw "SyncIntervalSeconds must be at least 15" }
if (-not $Zones -or $Zones.Count -eq 0) { throw "At least one zone is required" }
if ($DryRun -and $CreateInstances) { throw "DryRun never creates cloud instances" }
if ($PackageOnly -and $CreateInstances) { throw "PackageOnly never creates cloud instances" }
if ($ResumeExisting -and $PackageOnly) { throw "ResumeExisting cannot repackage a frozen run" }

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
function Resolve-RepoPath([string]$Value) {
    $candidate = if ([System.IO.Path]::IsPathRooted($Value)) { $Value } else { Join-Path $repoRoot $Value }
    return (Resolve-Path -LiteralPath $candidate).Path
}

$runDir = Join-Path $repoRoot "outputs/gcp_runs/$RunName"
$packageDir = Join-Path $runDir "package"
$sourcePath = Join-Path $runDir "ofc_regular_hu_m43_attempt03_population_source.zip"
$manifestPath = Join-Path $runDir "population_run_manifest.json"
$shardsPath = Join-Path $runDir "population_shards.jsonl"
$startupPath = Join-Path $runDir "startup_hu_m43_attempt03_population.sh"
$gcsPrefix = "gs://$Bucket/runs/$RunName"
$sourceUri = "$gcsPrefix/source/ofc_regular_hu_m43_attempt03_population_source.zip"
$manifestUri = "$gcsPrefix/source/population_run_manifest.json"
$shardsUri = "$gcsPrefix/source/population_shards.jsonl"
$startupUri = "$gcsPrefix/source/startup_hu_m43_attempt03_population.sh"
$planUri = "$gcsPrefix/source/population_plan.json"
$vmPrefix = Convert-ToVmPrefix $RunName

if ($ResumeExisting) {
    foreach ($path in @($manifestPath, $sourcePath, $shardsPath, $startupPath)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Frozen Attempt03 population input is missing: $path" }
    }
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    if ($manifest.schema -ne $ManifestSchema -or $manifest.run_name -ne $RunName -or
        $manifest.project_id -ne $ProjectId -or $manifest.bucket -ne $Bucket -or
        $manifest.runtime.model_schema -ne $ModelSchema -or
        $manifest.runtime.action_score_mode -ne $ActionScoreMode -or
        $manifest.launch_preflight.schema -ne $PreflightSchema -or
        $manifest.launch_preflight.status -ne "pass" -or
        $manifest.source_boundary.runtime_artifacts_only -ne $true -or
        $manifest.no_runtime_activation -ne $true -or $manifest.current_profile_mutated -ne $false) {
        throw "Frozen Attempt03 population run identity mismatch"
    }
    foreach ($entry in @(
        @($sourcePath, $manifest.source.sha256),
        @($shardsPath, $manifest.shards.sha256),
        @($startupPath, $manifest.startup.sha256)
    )) {
        Assert-Sha256Text $entry[1] "frozen manifest hash"
        if ((Get-Sha256 $entry[0]) -ne $entry[1]) { throw "Frozen Attempt03 local hash chain is broken" }
    }
    if (-not $DryRun) {
        foreach ($uri in @($manifestUri, $sourceUri, $shardsUri, $startupUri, $planUri)) {
            if (-not (Test-GcsObject $uri)) { throw "Frozen remote Attempt03 object is missing: $uri" }
        }
    }
}
else {
    if (Test-Path -LiteralPath $runDir) { throw "Attempt03 population run directory already exists and is immutable: $runDir" }
    $resolved = [ordered]@{
        model = Resolve-RepoPath $ModelPath
        final_training_manifest = Resolve-RepoPath $FinalTrainingManifestPath
        runtime_freeze = Resolve-RepoPath $RuntimeFreezePath
        training_freeze = Resolve-RepoPath $TrainingFreezePath
        precalibration_receipt = Resolve-RepoPath $PrecalibrationReceiptPath
        model_freeze = Resolve-RepoPath $ModelFreezePath
        attempt03_plan = Resolve-RepoPath $Attempt03PlanPath
        locked_receipt = Resolve-RepoPath $LockedReceiptPath
        consumption_marker = Resolve-RepoPath $ConsumptionMarkerPath
        population_plan = Resolve-RepoPath $PlanPath
    }
    $preflightArgs = @(
        "-m", "ofc_regular.validate_hu_m43_attempt03_population", "preflight",
        "--model", $resolved.model,
        "--final-training-manifest", $resolved.final_training_manifest,
        "--runtime-freeze", $resolved.runtime_freeze,
        "--training-freeze", $resolved.training_freeze,
        "--precalibration-receipt", $resolved.precalibration_receipt,
        "--model-freeze", $resolved.model_freeze,
        "--attempt03-plan", $resolved.attempt03_plan,
        "--locked-receipt", $resolved.locked_receipt,
        "--consumption-marker", $resolved.consumption_marker,
        "--population-plan", $resolved.population_plan
    )
    $oldPythonPath = $env:PYTHONPATH
    $oldPreference = $ErrorActionPreference
    $env:PYTHONPATH = Join-Path $repoRoot "src"
    $ErrorActionPreference = "Continue"
    try { $preflightRaw = @(& python @preflightArgs 2>&1); $preflightCode = $LASTEXITCODE }
    finally { $env:PYTHONPATH = $oldPythonPath; $ErrorActionPreference = $oldPreference }
    if ($preflightCode -ne 0) { throw "Attempt03 population preflight failed ($preflightCode):`n$(@($preflightRaw) -join [Environment]::NewLine)" }
    $preflight = (@($preflightRaw) -join "`n") | ConvertFrom-Json
    if ($preflight.schema -ne $PreflightSchema -or $preflight.status -ne "pass" -or
        $preflight.model_schema -ne $ModelSchema -or $preflight.action_score_mode -ne $ActionScoreMode -or
        $preflight.canonical_global_marker_verified -ne $true -or
        $preflight.teacher_calibration_locked_content_packaged -ne $false -or
        [int]$preflight.teacher_overlap_count -ne 0 -or [int]$preflight.prior_population_overlap_count -ne 0 -or
        $preflight.current_profile_mutated -ne $false -or $preflight.no_runtime_activation -ne $true) {
        throw "Attempt03 population preflight returned an unsafe receipt"
    }
    $plan = Get-Content -LiteralPath $resolved.population_plan -Raw | ConvertFrom-Json

    New-Item -ItemType Directory -Path $packageDir -Force | Out-Null
    foreach ($item in @("pyproject.toml", "src")) {
        Copy-Item -LiteralPath (Join-Path $repoRoot $item) -Destination $packageDir -Recurse
    }
    Get-ChildItem -LiteralPath (Join-Path $packageDir "src") -Recurse -File -Filter "*.pyc" | Remove-Item -Force
    Get-ChildItem -LiteralPath (Join-Path $packageDir "src") -Recurse -Directory -Filter "__pycache__" |
        Sort-Object FullName -Descending | Remove-Item -Recurse -Force
    $postCopyV5Verifier = @'
import hashlib,json,pathlib,sys
package=pathlib.Path(sys.argv[1]).resolve()
freeze=json.load(open(sys.argv[2],encoding="utf-8-sig"))
expected=freeze["executable_model_freeze"]["v5_implementation_sha256"]
if not isinstance(expected,str) or len(expected)!=64 or any(c not in "0123456789abcdef" for c in expected):
 raise ValueError("runtime freeze v5 implementation SHA is invalid")
source=package/"src/ofc_regular/hu_m43_joint_model_v5.py"
if not source.is_file(): raise ValueError("packaged v5 implementation is missing")
actual=hashlib.sha256(source.read_bytes()).hexdigest()
if actual!=expected: raise ValueError(f"packaged v5 implementation SHA changed: {actual} != {expected}")
print(json.dumps({"schema":"hu_m43_attempt03_population_post_copy_source_audit_v1","status":"pass","v5_implementation_sha256":actual},sort_keys=True))
'@
    $oldPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $postCopyV5Raw = @($postCopyV5Verifier | & python - $packageDir $resolved.runtime_freeze 2>&1)
        $postCopyV5Code = $LASTEXITCODE
    }
    finally { $ErrorActionPreference = $oldPreference }
    if ($postCopyV5Code -ne 0) {
        throw "Attempt03 packaged v5 post-copy verification failed ($postCopyV5Code):`n$(@($postCopyV5Raw) -join [Environment]::NewLine)"
    }
    $postCopyV5Audit = (@($postCopyV5Raw) -join "`n") | ConvertFrom-Json
    if ($postCopyV5Audit.schema -ne "hu_m43_attempt03_population_post_copy_source_audit_v1" -or
        $postCopyV5Audit.status -ne "pass") {
        throw "Attempt03 packaged v5 post-copy audit schema changed"
    }
    $dependencyRows = @()
    foreach ($relative in $RequiredModels) {
        $source = Join-Path $repoRoot $relative
        if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { throw "Required population model is missing: $relative" }
        $destination = Join-Path $packageDir $relative
        New-Item -ItemType Directory -Path (Split-Path $destination -Parent) -Force | Out-Null
        Copy-Item -LiteralPath $source -Destination $destination
        $dependencyRows += [ordered]@{ path = $relative; bytes = [long](Get-Item $source).Length; sha256 = Get-Sha256 $source }
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
    Copy-Item -LiteralPath $resolved.final_training_manifest -Destination (Join-Path $artifactDir "final_training_manifest.json")
    Copy-Item -LiteralPath $resolved.runtime_freeze -Destination (Join-Path $artifactDir "runtime_freeze.json")
    Copy-Item -LiteralPath $resolved.population_plan -Destination (Join-Path $artifactDir "population_plan.json")
    $artifactFiles = @(Get-ChildItem -LiteralPath $artifactDir -File | Select-Object -ExpandProperty Name | Sort-Object)
    if (($artifactFiles -join ',') -ne 'final_training_manifest.json,m43_model.pkl,population_plan.json,runtime_freeze.json') {
        throw "Attempt03 package must contain runtime artifacts only; teacher-valued receipts and holdouts are forbidden"
    }
    if (Get-ChildItem -LiteralPath $packageDir -Recurse -File | Where-Object { $_.Extension -eq ".jsonl" }) {
        throw "Teacher/calibration/locked JSONL must not enter the Attempt03 population package"
    }
    $artifactHashes = [ordered]@{
        model = Get-Sha256 (Join-Path $artifactDir "m43_model.pkl")
        final_training_manifest = Get-Sha256 (Join-Path $artifactDir "final_training_manifest.json")
        runtime_freeze = Get-Sha256 (Join-Path $artifactDir "runtime_freeze.json")
        population_plan = Get-Sha256 (Join-Path $artifactDir "population_plan.json")
    }
    foreach ($binding in @(
        @("model", "model_sha256"),
        @("final_training_manifest", "final_training_manifest_file_sha256"),
        @("runtime_freeze", "runtime_freeze_file_sha256"),
        @("population_plan", "population_plan_file_sha256")
    )) {
        if ($artifactHashes[$binding[0]] -ne $preflight.($binding[1])) { throw "Attempt03 lifecycle artifact changed during packaging: $($binding[0])" }
    }

    $shards = @()
    for ($index = 0; $index -lt [int]$plan.shards; $index++) {
        $offset = $index * [int]$plan.paired_seeds_per_shard
        $shards += [ordered]@{
            shard = $index
            offset = $offset
            seed = [long]$plan.seed + [long]$offset * [long]$plan.seed_stride
            seed_stride = [long]$plan.seed_stride
            paired_seeds = [int]$plan.paired_seeds_per_shard
            output_prefix = ("shard-{0:D4}" -f $index)
        }
    }
    Write-Utf8NoBom $shardsPath ((@($shards | ForEach-Object { $_ | ConvertTo-Json -Compress }) -join "`n") + "`n")
    Copy-Item -LiteralPath $shardsPath -Destination (Join-Path $packageDir "population_shards.jsonl")
    Write-Utf8NoBom (Join-Path $packageDir "population_source_models.json") (($dependencyRows | ConvertTo-Json -Depth 5) + "`n")
    if (Test-Path -LiteralPath (Join-Path $packageDir "configs")) {
        throw "Repository configs/current-profile metadata must not enter the Attempt03 population package"
    }
    if (Test-Path -LiteralPath (Join-Path $packageDir "rust")) {
        throw "Rust source/debug artifacts must not enter the Attempt03 population package"
    }
    $allowedJsonl = [System.IO.Path]::GetFullPath((Join-Path $packageDir "population_shards.jsonl"))
    $unexpectedJsonl = @(Get-ChildItem -LiteralPath $packageDir -Recurse -File | Where-Object {
        $_.Extension -ieq ".jsonl" -and $_.FullName -cne $allowedJsonl
    })
    if ($unexpectedJsonl.Count) { throw "Teacher/calibration/locked JSONL found outside the sole shard-spec allowlist entry" }
    New-ZipWithForwardSlashes $packageDir $sourcePath

    $startup = @'
#!/usr/bin/env bash
set -Eeuo pipefail
LOG=/var/log/hu_m43_attempt03_population.log
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
json.dump({'schema':'hu_m43_attempt03_population_spot_status_v1','run_name':run,'shard':int(shard),'state':state,'exit_code':int(code),'instance':instance,'zone':zone,'manifest_sha256':manifest,'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat()},open(p,'w'),sort_keys=True)
PY
gcloud storage cp "$STATUS" "$PREFIX/status/shard-${SHARD}.json" >/dev/null || true; }
cleanup(){ code=$?; [[ -z "$HEARTBEAT_PID" ]] || kill "$HEARTBEAT_PID" >/dev/null 2>&1 || true; if [[ $code -ne 0 ]]; then write_status failed "$code"; gcloud storage cp "$LOG" "$PREFIX/results/shard-$(printf '%04d' "$SHARD")/startup.log" >/dev/null || true; fi; if [[ "$SELF_DELETE" == 1 ]]; then gcloud compute instances delete "$INSTANCE" --zone "$ZONE" --quiet >/dev/null 2>&1 || true; fi; exit "$code"; }
trap cleanup EXIT
write_status booting 0
RESULT="$PREFIX/results/shard-$(printf '%04d' "$SHARD")"
if gcloud storage ls "$RESULT/DONE" >/dev/null 2>&1; then write_status complete 0; exit 0; fi
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
for k,v in {'SEED':x['seed'],'STRIDE':x['seed_stride'],'COUNT':x['paired_seeds']}.items(): print(f'{k}={shlex.quote(str(v))}')
PY
)"
python3 -m venv .venv; source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'numpy==2.2.6' 'scikit-learn==1.8.0' 'lightgbm==4.6.0'
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
python -m ofc_regular.evaluate_hu_m4_population --model artifacts/m43_model.pkl --expected-model-sha256 "$MODEL_SHA" --freeze-manifest artifacts/runtime_freeze.json --training-manifest artifacts/final_training_manifest.json --paired-seeds "$COUNT" --seed "$SEED" --seed-stride "$STRIDE" --opponents stage19_p0 stage9f_p2 stage7_m5_r10 random_exact_final --records-output "$OUT/records.jsonl" --output "$OUT/evaluation.json" --progress-every 10 > "$OUT/evaluation_stdout.log"
kill "$HEARTBEAT_PID" >/dev/null 2>&1 || true; HEARTBEAT_PID=""
python - "$OUT/evaluation.json" "$OUT/records.jsonl" "$SEED" "$STRIDE" "$COUNT" "$MODEL_SHA" <<'PY'
import json,sys
e=json.load(open(sys.argv[1])); rows=[json.loads(x) for x in open(sys.argv[2]) if x.strip()]
seed,stride,count=int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5]); model=sys.argv[6]
assert e['schema']=='hu_m4_t1_population_evaluation_v1' and e['paired_seat_swap'] is True
assert e['seed']==seed and e['seed_stride']==stride and e['paired_seeds_per_opponent']==count
assert e['opponents']==['stage19_p0','stage9f_p2','stage7_m5_r10','random_exact_final']
assert len(rows)==count*8 and e['trace_hands']==count*16
assert all(row.get('runtime_binding_verified') is True for row in rows)
r=e['runtime_config']
assert r['current_profile_used'] is False and r['promotion_artifact_contract'] is True and r['diagnostic_legacy'] is False
assert r['candidate_model_sha256']==model and r['safety_model_sha256']==model
assert r['model_schema']=='hu_m43_t1_joint_model_v5' and r['runtime_binding_verified'] is True
assert r['action_score_mode']=='eligible_stage18_stacked_meta_ranker_v5' and r['safety_enabled'] is True
PY
EVAL_SHA="$(sha256sum "$OUT/evaluation.json" | cut -d' ' -f1)"; RECORDS_SHA="$(sha256sum "$OUT/records.jsonl" | cut -d' ' -f1)"
python - "$OUT/DONE" "$RUN_NAME" "$SHARD" "$MANIFEST_SHA" "$SOURCE_SHA" "$MODEL_SHA" "$EVAL_SHA" "$RECORDS_SHA" <<'PY'
import json,sys
p,run,shard,manifest,source,model,evaluation,records=sys.argv[1:]
json.dump({'schema':'hu_m43_attempt03_population_spot_done_v1','status':'complete','run_name':run,'shard':int(shard),'manifest_sha256':manifest,'source_sha256':source,'model_sha256':model,'evaluation_sha256':evaluation,'records_sha256':records,'current_profile_mutated':False,'no_runtime_activation':True},open(p,'w'),sort_keys=True)
PY
gcloud storage cp "$OUT/evaluation.json" "$RESULT/evaluation.json" >/dev/null
gcloud storage cp "$OUT/records.jsonl" "$RESULT/records.jsonl" >/dev/null
gcloud storage cp "$OUT/evaluation_stdout.log" "$RESULT/evaluation_stdout.log" >/dev/null
gcloud storage cp "$LOG" "$RESULT/startup.log" >/dev/null
write_status complete 0
gcloud storage cp "$OUT/DONE" "$RESULT/DONE" --if-generation-match=0 >/dev/null
'@
    Write-Utf8NoBom $startupPath ($startup + "`n")

    $manifest = [ordered]@{
        schema = $ManifestSchema
        run_name = $RunName
        project_id = $ProjectId
        bucket = $Bucket
        population_plan = [ordered]@{ uri = $planUri; sha256 = [string]$preflight.population_plan_file_sha256; paired_seeds = [int]$plan.paired_seeds_per_opponent; seed = [long]$plan.seed; seed_stride = [long]$plan.seed_stride; shards = [int]$plan.shards; paired_seeds_per_shard = [int]$plan.paired_seeds_per_shard }
        source = [ordered]@{ uri = $sourceUri; sha256 = Get-Sha256 $sourcePath; bytes = [long](Get-Item $sourcePath).Length }
        startup = [ordered]@{ uri = $startupUri; sha256 = Get-Sha256 $startupPath }
        shards = [ordered]@{ uri = $shardsUri; sha256 = Get-Sha256 $shardsPath; count = [int]$plan.shards }
        runtime = [ordered]@{
            model_sha256 = [string]$preflight.model_sha256
            model_schema = $ModelSchema
            model_id = [string]$preflight.model_id
            action_score_mode = $ActionScoreMode
            frozen_threshold = [double]$preflight.frozen_threshold
            training_manifest_sha256 = [string]$preflight.final_training_manifest_file_sha256
            training_freeze_sha256 = [string]$preflight.training_freeze_file_sha256
            freeze_manifest_sha256 = [string]$preflight.runtime_freeze_file_sha256
            locked_receipt_sha256 = [string]$preflight.locked_receipt_file_sha256
            consumption_marker_sha256 = [string]$preflight.consumption_marker_file_sha256
            runtime_teacher_inputs = $false
            current_profile_used = $false
        }
        launch_preflight = $preflight
        source_boundary = [ordered]@{
            teacher_jsonl_packaged = $false
            calibration_jsonl_packaged = $false
            locked_jsonl_packaged = $false
            teacher_valued_receipts_packaged = $false
            current_profile_artifact_packaged = $false
            runtime_artifacts_only = $true
        }
        compute = [ordered]@{ machine_type = $MachineType; fallback_machine_types = $FallbackMachineTypes; zones = $Zones; boot_disk_gb = $BootDiskGb; provisioning_model = "SPOT"; instance_termination_action = "DELETE"; self_delete = (-not [bool]$NoSelfDelete); sync_interval_seconds = $SyncIntervalSeconds }
        checkpoint = [ordered]@{ unit = "completed_shard"; retry = "deterministic_full_shard"; resume_missing_shards_only = $true; done_commit_last = $true }
        no_runtime_activation = $true
        current_profile_mutated = $false
    }
    Write-Utf8NoBom $manifestPath (($manifest | ConvertTo-Json -Depth 15) + "`n")
}

$manifestSha = Get-Sha256 $manifestPath
if ($PackageOnly -or $DryRun) {
    [ordered]@{ schema = "hu_m43_attempt03_population_package_result_v1"; run_name = $RunName; mode = $(if ($DryRun) { "dry_run" } else { "package_only" }); run_dir = $runDir; manifest_sha256 = $manifestSha; source_sha256 = Get-Sha256 $sourcePath; shards = [int]$manifest.shards.count; no_runtime_activation = $true; current_profile_mutated = $false } | ConvertTo-Json -Depth 6
    exit 0
}

if (-not $ResumeExisting) {
    foreach ($pair in @(
        @($sourcePath, $sourceUri), @($startupPath, $startupUri),
        @($shardsPath, $shardsUri), @((Join-Path $packageDir "artifacts/population_plan.json"), $planUri)
    )) {
        if (Test-GcsObject $pair[1]) { throw "Immutable Attempt03 population object already exists: $($pair[1])" }
        Invoke-Gcloud @("storage", "cp", $pair[0], $pair[1], "--project", $ProjectId, "--if-generation-match=0") | Out-Null
    }
    if (Test-GcsObject $manifestUri) { throw "Immutable Attempt03 population manifest already exists: $manifestUri" }
    Invoke-Gcloud @("storage", "cp", $manifestPath, $manifestUri, "--project", $ProjectId, "--if-generation-match=0") | Out-Null
}
if (-not $CreateInstances) {
    [ordered]@{ run_name = $RunName; state = "packaged_and_uploaded"; manifest_sha256 = $manifestSha; create_instances = $false; no_runtime_activation = $true } | ConvertTo-Json -Depth 6
    exit 0
}

$selected = if ($StartShards.Count) { @($StartShards | ForEach-Object { [int]$_ }) } else { @(0..([int]$manifest.shards.count - 1)) }
if (@($selected | Select-Object -Unique).Count -ne $selected.Count) { throw "Attempt03 population shard selection contains duplicates" }
if (@($selected | Where-Object { $_ -ne 0 }).Count) {
    $canaryUri = "$gcsPrefix/results/shard-0000/DONE"
    if (-not (Test-GcsObject $canaryUri)) { throw "Attempt03 population fanout requires a completed shard-0000 canary" }
    $canary = ((Invoke-Gcloud @("storage", "cat", $canaryUri, "--project", $ProjectId)) -join "`n") | ConvertFrom-Json
    if ($canary.schema -ne $DoneSchema -or $canary.status -ne "complete" -or
        $canary.run_name -ne $RunName -or [int]$canary.shard -ne 0 -or
        $canary.manifest_sha256 -ne $manifestSha -or
        $canary.source_sha256 -ne $manifest.source.sha256 -or
        $canary.model_sha256 -ne $manifest.runtime.model_sha256 -or
        $canary.current_profile_mutated -ne $false -or $canary.no_runtime_activation -ne $true) {
        throw "Attempt03 population fanout canary is not bound to the frozen run"
    }
}
foreach ($shard in $selected) {
    if ($shard -lt 0 -or $shard -ge [int]$manifest.shards.count) { throw "Shard index outside frozen Attempt03 plan: $shard" }
    $resultPrefix = "shard-{0:D4}" -f $shard
    if (Test-GcsObject "$gcsPrefix/results/$resultPrefix/DONE") { continue }
    $vmName = "$vmPrefix-s{0:D3}" -f $shard
    $existing = Invoke-Gcloud @("compute", "instances", "list", "--project", $ProjectId, "--filter", "name=$vmName", "--format=value(name)")
    if (@($existing | Where-Object { $_ }).Count) {
        if ($SkipExistingInstances) { continue }
        throw "Attempt03 population worker already exists: $vmName"
    }
    $created = $false; $errors = @(); $attempt = 0
    foreach ($machine in @($MachineType) + @($FallbackMachineTypes)) {
        $zone = $Zones[$attempt % $Zones.Count]; $attempt++
        $arguments = @("compute", "instances", "create", $vmName, "--project", $ProjectId, "--zone", $zone, "--machine-type", $machine, "--provisioning-model", "SPOT", "--instance-termination-action", "DELETE", "--boot-disk-size", "${BootDiskGb}GB", "--image-family", "debian-12", "--image-project", "debian-cloud", "--scopes", "cloud-platform", "--metadata", "RUN_NAME=$RunName,BUCKET=$Bucket,SHARD=$shard,SOURCE_URI=$($manifest.source.uri),SOURCE_SHA=$($manifest.source.sha256),MANIFEST_SHA=$manifestSha,SYNC_SECONDS=$SyncIntervalSeconds,SELF_DELETE=$(if ($NoSelfDelete) { 0 } else { 1 })", "--metadata-from-file", "startup-script=$startupPath")
        try { Invoke-Gcloud $arguments | Out-Null; $created = $true; break }
        catch { $errors += $_.Exception.Message }
    }
    if (-not $created) { throw "Could not create Attempt03 population worker $vmName`n$($errors -join [Environment]::NewLine)" }
}
[ordered]@{ run_name = $RunName; state = "workers_started"; shards = $selected; manifest_sha256 = $manifestSha; provisioning_model = "SPOT"; no_runtime_activation = $true; current_profile_mutated = $false } | ConvertTo-Json -Depth 8
