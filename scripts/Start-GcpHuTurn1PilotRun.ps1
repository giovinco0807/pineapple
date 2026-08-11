param(
    [string]$ProjectId = "ofc-solver-485418",
    [string]$Bucket = "pokerhu-ofc-solver-485418-training",
    [string]$RunName = ("regular-hu-t1-stage1-pilot-" + (Get-Date -Format "yyyyMMdd-HHmmss")),
    [int]$TotalSamples = 10,
    [int]$ShardSamples = 1,
    [int]$RecordSkipBase = 0,
    [int]$RecordSkipStride = 0,
    [switch]$FastSkipRecords,
    [int]$BaseSeed = 2026062401,
    [int]$SeedStride = 1000000,
    [int]$VmCount = 10,
    [string]$MachineType = "e2-highcpu-4",
    [int]$CpuThreads = 1,
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
    [int]$FutureSamples = 1,
    [int]$MaxActions = 0,
    [int]$OpeningLookaheadSamples = 1,
    [string]$Profile = "stage9f_p2",
    [string]$OpponentProfile = "stage9f_p2",
    [string]$SourceBucket = "natural_mc32",
    [string[]]$Seats = @("first", "second"),
    [string]$CandidateModel = "",
    [string[]]$CandidateModels = @(),
    [int]$CandidateTopk = 0,
    [int]$CandidateUnionCap = 0,
    [ValidateSet("min_rank", "rank_sum", "reciprocal_rank_sum", "mean_score", "max_score", "mean_z_score", "max_z_score")]
    [string]$CandidateUnionMode = "min_rank",
    [string]$HuTurn2Stage8bModel = "models/hu_turn2_stage8b_safe_lcb196_20k_reference_override_cached_rank_wide.pt",
    [string]$TeacherModule = "ofc_regular.hu_turn1_teacher_pilot",
    [string]$PhaseName = "hu_t1_stage1_pilot",
    [string[]]$AdditionalRequiredModels = @(),
    [string[]]$AdditionalPackageFiles = @(),
    [int]$BootDiskGb = 50,
    [string[]]$StartShards = @(),
    [switch]$CollectTopkLog,
    [switch]$CreateInstances,
    [switch]$SkipExistingInstances,
    [switch]$NoSpot,
    [switch]$NoSelfDelete,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$cloudSdkGcloud = Join-Path $env:LOCALAPPDATA "Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
if (Test-Path $cloudSdkGcloud) {
    Set-Alias -Name gcloud -Value $cloudSdkGcloud -Scope Script
}
elseif (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    throw "gcloud not found in PATH or at $cloudSdkGcloud"
}

if ($TotalSamples -le 0) { throw "TotalSamples must be positive" }
if ($ShardSamples -le 0) { throw "ShardSamples must be positive" }
if ($RecordSkipBase -lt 0) { throw "RecordSkipBase must be non-negative" }
if ($RecordSkipStride -lt 0) { throw "RecordSkipStride must be non-negative" }
if ($VmCount -le 0) { throw "VmCount must be positive" }
if ($CpuThreads -le 0) { throw "CpuThreads must be positive" }
if ($FutureSamples -le 0) { throw "FutureSamples must be positive" }
if ($MaxActions -lt 0) { throw "MaxActions must be non-negative" }
if ($CandidateTopk -lt 0) { throw "CandidateTopk must be non-negative" }
if ($CandidateUnionCap -lt 0) { throw "CandidateUnionCap must be non-negative" }
if ($CandidateModel -and $CandidateModels.Count -gt 0) { throw "CandidateModel and CandidateModels are mutually exclusive" }
if ($CandidateModel -and $CandidateTopk -le 0) { throw "CandidateTopk must be positive when CandidateModel is set" }
if ($CandidateModels.Count -gt 0 -and $CandidateTopk -le 0) { throw "CandidateTopk must be positive when CandidateModels is set" }
if ($OpeningLookaheadSamples -lt 0) { throw "OpeningLookaheadSamples must be non-negative" }
if (-not $Seats -or @($Seats | Where-Object { $_ -notin @("first", "second") }).Count -gt 0) {
    throw "Seats must contain first and/or second"
}
$Seats = @($Seats | Select-Object -Unique)
if (-not $Zones -or $Zones.Count -eq 0) { throw "At least one zone is required" }
if ($RunName -notmatch '^[a-zA-Z0-9][a-zA-Z0-9._-]*$') {
    throw "RunName may only contain letters, numbers, dot, underscore, and dash"
}
if ($TeacherModule -notmatch '^[a-zA-Z_][a-zA-Z0-9_.]*$') {
    throw "TeacherModule must be a Python module path"
}
if ($PhaseName -notmatch '^[a-zA-Z0-9_][a-zA-Z0-9_.-]*$') {
    throw "PhaseName contains unsupported characters"
}
if ($Seats.Count -gt 1 -and $ShardSamples -eq 1 -and $RecordSkipBase -eq 0 -and $RecordSkipStride -eq 0) {
    Write-Warning (
        "ShardSamples=1 with no record skip emits only the first T1 row per generated hand " +
        "(usually player0/first). Use ShardSamples=2 for paired first+second rows, " +
        "or launch a second-only complement with RecordSkipBase=1 and FastSkipRecords."
    )
}

function Convert-ToVmName {
    param([string]$Name)
    $vmName = $Name.ToLowerInvariant() -replace '[^a-z0-9-]', '-'
    $vmName = $vmName.Trim('-')
    if ($vmName.Length -gt 54) { $vmName = $vmName.Substring(0, 54).Trim('-') }
    if (-not $vmName) { throw "RunName does not produce a valid VM name" }
    return $vmName
}

function Write-Utf8NoBom {
    param([string]$Path, [string]$Text)
    $encoding = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::WriteAllText($Path, $Text, $encoding)
}

function New-ZipWithForwardSlashes {
    param([string]$SourceDir, [string]$DestinationPath)
    Add-Type -AssemblyName System.IO.Compression
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    if (Test-Path $DestinationPath) { Remove-Item -LiteralPath $DestinationPath -Force }
    $sourceRoot = (Resolve-Path -LiteralPath $SourceDir).Path.TrimEnd('\', '/')
    $prefixLength = $sourceRoot.Length + 1
    $zip = [System.IO.Compression.ZipFile]::Open($DestinationPath, [System.IO.Compression.ZipArchiveMode]::Create)
    try {
        Get-ChildItem -LiteralPath $sourceRoot -Recurse -File | ForEach-Object {
            $relativePath = $_.FullName.Substring($prefixLength)
            $entryName = $relativePath -replace '\\', '/'
            [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
                $zip,
                $_.FullName,
                $entryName,
                [System.IO.Compression.CompressionLevel]::Optimal
            ) | Out-Null
        }
    }
    finally {
        $zip.Dispose()
    }
}

$repoRoot = (Get-Location).Path
$requiredModels = @(
    "models/opening_stage7_torch_wide.pt",
    "models/turn1_stage6_torch_wide.pt",
    "models/turn2_stage8.pkl",
    "models/turn3_stage6.pkl",
    "models/hu_turn3_stage7_reference_override_cached_rank_wide.pt",
    "models/hu_turn3_stage3_mc32_500k_plus_m8_12_f128_100k_w2_cached_rank_wide.pt",
    $HuTurn2Stage8bModel
) | Select-Object -Unique
if ($AdditionalRequiredModels.Count -gt 0) {
    $requiredModels += $AdditionalRequiredModels
    $requiredModels = @($requiredModels | Select-Object -Unique)
}
if ($CandidateModel) {
    $requiredModels += $CandidateModel
    $requiredModels = @($requiredModels | Select-Object -Unique)
}
if ($CandidateModels.Count -gt 0) {
    $requiredModels += $CandidateModels
    $requiredModels = @($requiredModels | Select-Object -Unique)
}
foreach ($model in $requiredModels) {
    if ([System.IO.Path]::IsPathRooted($model)) {
        throw "Model paths must be repo-relative: $model"
    }
    if (-not (Test-Path (Join-Path $repoRoot $model))) {
        throw "Required model not found: $model"
    }
}
$additionalPackageEntries = @()
foreach ($relativePath in $AdditionalPackageFiles) {
    if ([System.IO.Path]::IsPathRooted($relativePath)) {
        throw "Additional package files must be repo-relative: $relativePath"
    }
    $sourcePath = Join-Path $repoRoot $relativePath
    if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
        throw "Additional package file not found: $relativePath"
    }
    $destinationName = Split-Path $relativePath -Leaf
    if ($additionalPackageEntries.destination_name -contains $destinationName) {
        throw "Additional package file names must be unique: $destinationName"
    }
    $additionalPackageEntries += [pscustomobject]@{
        source = $relativePath
        destination_name = $destinationName
        package_path = ("inputs/{0}" -f $destinationName)
    }
}

$totalShards = [int][Math]::Ceiling($TotalSamples / [double]$ShardSamples)
$StartShardIndices = @()
foreach ($startShardValue in $StartShards) {
    foreach ($part in ($startShardValue -split ",")) {
        $trimmed = $part.Trim()
        if ($trimmed) { $StartShardIndices += [int]$trimmed }
    }
}
$StartShardIndices = @($StartShardIndices | Sort-Object -Unique)
foreach ($startShard in $StartShardIndices) {
    if ($startShard -lt 0 -or $startShard -ge $totalShards) {
        throw "StartShards values must be in [0, total_shards): $startShard"
    }
}

$shardSpecs = New-Object System.Collections.Generic.List[object]
for ($shardIndex = 0; $shardIndex -lt $totalShards; $shardIndex += 1) {
    $samplesDone = $shardIndex * $ShardSamples
    $samples = [Math]::Min($ShardSamples, $TotalSamples - $samplesDone)
    $seed = $BaseSeed + ($shardIndex * $SeedStride)
    $skipRecords = $RecordSkipBase + ($shardIndex * $RecordSkipStride)
    $shardSpecs.Add([ordered]@{
        shard = $shardIndex
        samples = $samples
        seed = $seed
        skip_records = $skipRecords
        output_prefix = ("shard_{0:D3}_seed{1}_samples{2:D05}" -f $shardIndex, $seed, $samples)
    })
}

$largeTeacherStatus = "No-Go"
if ($Profile -eq "stage9f_p2" -and $OpponentProfile -eq "stage9f_p2") {
    $largeTeacherStatus = "Go for clean high-MC T1 teacher pilot; production-scale T1 remains No-Go"
}
elseif ($Profile -eq "stage9f_fast_t2_t1_teacher" -and $OpponentProfile -eq "stage9f_fast_t2_t1_teacher") {
    $largeTeacherStatus = "Go for fast diagnostic T1 teacher pilot only; production-scale T1 remains No-Go"
}
elseif ($PhaseName -eq "hu_t0_stage1_pilot" -and $Profile -eq "stage18_p1" -and $OpponentProfile -eq "stage18_p1") {
    $largeTeacherStatus = "Go for HU T0 all-action pilot; production-scale T0 remains No-Go"
}

if ($DryRun) {
    [pscustomobject]@{
        execution = "dry_run"
        run_name = $RunName
        project_id = $ProjectId
        bucket = $Bucket
        phase = $PhaseName
        teacher_module = $TeacherModule
        profile = $Profile
        opponent_profile = $OpponentProfile
        source_bucket = $SourceBucket
        seats = $Seats
        total_samples = $TotalSamples
        shard_samples = $ShardSamples
        total_shards = $totalShards
        vm_count = $VmCount
        machine_type = $MachineType
        cpu_threads = $CpuThreads
        zones = $Zones
        future_samples = $FutureSamples
        max_actions = $MaxActions
        opening_lookahead_samples = $OpeningLookaheadSamples
        candidate_model = $CandidateModel
        candidate_models = $CandidateModels
        candidate_topk = $CandidateTopk
        candidate_union_cap = $CandidateUnionCap
        candidate_union_mode = $CandidateUnionMode
        collect_topk_log = [bool]$CollectTopkLog
        create_instances = [bool]$CreateInstances
        production_default = "No-Go"
        large_teacher_status = $largeTeacherStatus
        additional_package_files = $additionalPackageEntries
        shards = $shardSpecs
    } | ConvertTo-Json -Depth 8
    exit 0
}

$runDir = Join-Path $repoRoot ("outputs/gcp_runs/{0}" -f $RunName)
$packageDir = Join-Path $runDir "package_src"
$packagePath = Join-Path $runDir "ofc_regular_hu_t1_stage1_pilot_source.zip"
$startupPath = Join-Path $runDir "startup_hu_t1_stage1_pilot_spot.sh"
$manifestPath = Join-Path $runDir "manifest.json"
$shardManifestPath = Join-Path $runDir "shards_manifest.jsonl"
$sourceUri = "gs://$Bucket/runs/$RunName/source/ofc_regular_hu_t1_stage1_pilot_source.zip"
$startupUri = "gs://$Bucket/runs/$RunName/source/startup_hu_t1_stage1_pilot_spot.sh"
$manifestUri = "gs://$Bucket/runs/$RunName/manifest.json"

New-Item -ItemType Directory -Force -Path $runDir | Out-Null
if (Test-Path $packageDir) { Remove-Item -LiteralPath $packageDir -Recurse -Force }
New-Item -ItemType Directory -Force -Path $packageDir | Out-Null

foreach ($item in @("pyproject.toml", "Cargo.toml", "Cargo.lock", "src", "rust", "configs")) {
    $source = Join-Path $repoRoot $item
    if (-not (Test-Path $source)) { throw "Missing package input: $item" }
    Copy-Item -LiteralPath $source -Destination $packageDir -Recurse
}
foreach ($model in $requiredModels) {
    $dest = Join-Path $packageDir $model
    New-Item -ItemType Directory -Force -Path (Split-Path $dest -Parent) | Out-Null
    Copy-Item -LiteralPath (Join-Path $repoRoot $model) -Destination $dest
}
foreach ($entry in $additionalPackageEntries) {
    $dest = Join-Path $packageDir $entry.package_path
    New-Item -ItemType Directory -Force -Path (Split-Path $dest -Parent) | Out-Null
    Copy-Item -LiteralPath (Join-Path $repoRoot $entry.source) -Destination $dest
}

$shardLines = @($shardSpecs | ForEach-Object { $_ | ConvertTo-Json -Compress -Depth 5 })
Write-Utf8NoBom -Path $shardManifestPath -Text (($shardLines -join "`n") + "`n")
Copy-Item -LiteralPath $shardManifestPath -Destination (Join-Path $packageDir "shards_manifest.jsonl")

New-ZipWithForwardSlashes -SourceDir $packageDir -DestinationPath $packagePath

$startup = @'
#!/usr/bin/env bash
set -euo pipefail
export HOME="${HOME:-/root}"
CURRENT_UID="$(id -u)"
CURRENT_GID="$(id -g)"

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
meta() { curl -fsS -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }
instance_meta() { curl -fsS -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/$1"; }
json_field() { python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get(sys.argv[2], ""))' "$1" "$2"; }

RUN_NAME="$(meta RUN_NAME)"
BUCKET="$(meta BUCKET)"
TOTAL_SHARDS="$(meta TOTAL_SHARDS)"
VM_COUNT="$(meta VM_COUNT)"
START_SHARD="$(meta START_SHARD)"
SOURCE_URI="$(meta SOURCE_URI)"
SELF_DELETE="$(meta SELF_DELETE)"
FUTURE_SAMPLES="$(meta FUTURE_SAMPLES)"
MAX_ACTIONS="$(meta MAX_ACTIONS)"
OPENING_LOOKAHEAD_SAMPLES="$(meta OPENING_LOOKAHEAD_SAMPLES)"
PROFILE="$(meta PROFILE)"
OPPONENT_PROFILE="$(meta OPPONENT_PROFILE)"
SOURCE_BUCKET="$(meta SOURCE_BUCKET)"
SEATS="$(meta SEATS)"
FAST_SKIP_RECORDS="$(meta FAST_SKIP_RECORDS)"
CANDIDATE_MODEL="$(meta CANDIDATE_MODEL)"
CANDIDATE_MODELS="$(meta CANDIDATE_MODELS)"
CANDIDATE_TOPK="$(meta CANDIDATE_TOPK)"
CANDIDATE_UNION_CAP="$(meta CANDIDATE_UNION_CAP)"
CANDIDATE_UNION_MODE="$(meta CANDIDATE_UNION_MODE)"
COLLECT_TOPK_LOG="$(meta COLLECT_TOPK_LOG)"
TEACHER_MODULE="$(meta TEACHER_MODULE)"
PHASE_NAME="$(meta PHASE_NAME)"
CPU_THREADS="$(meta CPU_THREADS)"
INSTANCE_NAME="$(instance_meta name)"
ZONE_PATH="$(instance_meta zone)"
ZONE="${ZONE_PATH##*/}"

cleanup_on_error() {
  code="$1"
  if [[ "$code" -eq 0 ]]; then return; fi
  log "startup failed exit=$code instance=$INSTANCE_NAME zone=$ZONE"
  if [[ "$SELF_DELETE" == "1" ]] && command -v gcloud >/dev/null 2>&1; then
    gcloud compute instances delete "$INSTANCE_NAME" --zone "$ZONE" --quiet || sudo shutdown -h now
  else
    sudo shutdown -h now || true
  fi
}
trap 'cleanup_on_error $?' EXIT

export DEBIAN_FRONTEND=noninteractive
export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS"

sudo apt-get update -y
sudo apt-get install -y unzip curl ca-certificates python3 python3-venv python3-pip

if ! command -v gcloud >/dev/null 2>&1; then
  sudo apt-get install -y apt-transport-https gnupg
  curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
  echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list >/dev/null
  sudo apt-get update -y
  sudo apt-get install -y google-cloud-cli
fi

if [[ ! -x "$HOME/.cargo/bin/rustup" ]]; then
  curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
fi
source "$HOME/.cargo/env"
rustup default stable

WORK_DIR="/opt/ofc-regular-hu-t1-stage1-pilot"
sudo rm -rf "$WORK_DIR"
sudo mkdir -p "$WORK_DIR"
sudo chown "$CURRENT_UID:$CURRENT_GID" "$WORK_DIR"
gcloud storage cp "$SOURCE_URI" /tmp/ofc_regular_hu_t1_stage1_pilot_source.zip
unzip -q /tmp/ofc_regular_hu_t1_stage1_pilot_source.zip -d "$WORK_DIR"
cd "$WORK_DIR"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "numpy==2.2.6" "scikit-learn==1.8.0"
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.6.0"
export PYTHONPATH="$WORK_DIR/src"

source "$HOME/.cargo/env"
cargo build --release
python - <<'PY'
from ofc_regular.hu_turn3_stage3_feature_rust import rust_direct_available
assert rust_direct_available(), "rust_direct encoder unavailable"
print("rust_direct_available=true")
PY

STATUS_DIR="/tmp/ofc-hu-t1-stage1-pilot-status"
RESULT_DIR="/tmp/ofc-hu-t1-stage1-pilot-results"
mkdir -p "$STATUS_DIR" "$RESULT_DIR"

log "start teacher pilot phase=$PHASE_NAME module=$TEACHER_MODULE run=$RUN_NAME start=$START_SHARD stride=$VM_COUNT total_shards=$TOTAL_SHARDS"
for (( shard=START_SHARD; shard<TOTAL_SHARDS; shard+=VM_COUNT )); do
  spec="$(sed -n "$((shard + 1))p" shards_manifest.jsonl)"
  if [[ -z "$spec" ]]; then
    log "missing spec for shard=$shard"
    exit 2
  fi
  samples="$(json_field "$spec" samples)"
  seed="$(json_field "$spec" seed)"
  skip_records="$(json_field "$spec" skip_records)"
  output_prefix="$(json_field "$spec" output_prefix)"
  remote_marker="gs://${BUCKET}/runs/${RUN_NAME}/results/${output_prefix}/DONE"
  status_remote="gs://${BUCKET}/runs/${RUN_NAME}/status/${output_prefix}.json"
  if gcloud storage ls "$remote_marker" >/dev/null 2>&1; then
    log "skip existing shard=$shard $output_prefix"
    continue
  fi

  local_out="${RESULT_DIR}/${output_prefix}"
  rm -rf "$local_out"
  mkdir -p "$local_out"
  log_path="${local_out}/run.log"
  status_path="${STATUS_DIR}/${output_prefix}.json"
  started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  start_seconds="$(date +%s)"
  log "run shard=$shard seed=$seed samples=$samples skip_records=$skip_records profile=$PROFILE opponent=$OPPONENT_PROFILE"

  args=(
    -B -m "$TEACHER_MODULE"
    --samples "$samples"
    --skip-records "$skip_records"
    --future-samples "$FUTURE_SAMPLES"
    --max-actions "$MAX_ACTIONS"
    --seed "$seed"
    --profile "$PROFILE"
    --opponent-profile "$OPPONENT_PROFILE"
    --source-bucket "$SOURCE_BUCKET"
    --opening-lookahead-samples "$OPENING_LOOKAHEAD_SAMPLES"
    --output "${local_out}/teacher.jsonl"
    --summary-output "${local_out}/summary.json"
  )
  IFS=';' read -r -a seat_values <<< "$SEATS"
  args+=(--seats "${seat_values[@]}")
  if [[ "$FAST_SKIP_RECORDS" == "1" ]]; then
    args+=(--fast-skip-records)
  fi
  if [[ -n "$CANDIDATE_MODEL" && "$CANDIDATE_TOPK" -gt 0 ]]; then
    args+=(--candidate-model "$CANDIDATE_MODEL" --candidate-topk "$CANDIDATE_TOPK")
  fi
  if [[ -n "$CANDIDATE_MODELS" && "$CANDIDATE_TOPK" -gt 0 ]]; then
    IFS=';' read -r -a candidate_model_paths <<< "$CANDIDATE_MODELS"
    args+=(--candidate-models)
    for candidate_model_path in "${candidate_model_paths[@]}"; do
      if [[ -n "$candidate_model_path" ]]; then
        args+=("$candidate_model_path")
      fi
    done
    args+=(--candidate-topk "$CANDIDATE_TOPK")
    if [[ "$CANDIDATE_UNION_CAP" -gt 0 ]]; then
      args+=(--candidate-union-cap "$CANDIDATE_UNION_CAP")
    fi
    args+=(--candidate-union-mode "$CANDIDATE_UNION_MODE")
  fi
  if [[ "$COLLECT_TOPK_LOG" == "1" ]]; then
    args+=(--collect-topk-log)
  fi

  set +e
  python "${args[@]}" > "$log_path" 2>&1
  exit_code=$?
  set -e

  finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  end_seconds="$(date +%s)"
  elapsed=$((end_seconds - start_seconds))
  if [[ "$exit_code" -eq 0 ]]; then
    printf 'complete\n' > "${local_out}/DONE"
    gcloud storage cp "$local_out"/* "gs://${BUCKET}/runs/${RUN_NAME}/results/${output_prefix}/" >/dev/null
    cat > "$status_path" <<EOF
{"run_name":"$RUN_NAME","phase":"$PHASE_NAME","teacher_module":"$TEACHER_MODULE","shard":$shard,"samples":$samples,"skip_records":$skip_records,"seed":$seed,"profile":"$PROFILE","opponent_profile":"$OPPONENT_PROFILE","status":"complete","instance":"$INSTANCE_NAME","zone":"$ZONE","elapsed_seconds":$elapsed,"started_at":"$started_at","finished_at":"$finished_at","output":"gs://${BUCKET}/runs/${RUN_NAME}/results/${output_prefix}"}
EOF
    gcloud storage cp "$status_path" "$status_remote" >/dev/null || true
  else
    gcloud storage cp "$log_path" "gs://${BUCKET}/runs/${RUN_NAME}/logs/${output_prefix}.log" >/dev/null || true
    cat > "$status_path" <<EOF
{"run_name":"$RUN_NAME","phase":"$PHASE_NAME","teacher_module":"$TEACHER_MODULE","shard":$shard,"samples":$samples,"skip_records":$skip_records,"seed":$seed,"profile":"$PROFILE","opponent_profile":"$OPPONENT_PROFILE","status":"failed","instance":"$INSTANCE_NAME","zone":"$ZONE","exit_code":$exit_code,"elapsed_seconds":$elapsed,"started_at":"$started_at","finished_at":"$finished_at","log":"gs://${BUCKET}/runs/${RUN_NAME}/logs/${output_prefix}.log"}
EOF
    gcloud storage cp "$status_path" "$status_remote" >/dev/null || true
    exit "$exit_code"
  fi
done

if [[ "$SELF_DELETE" == "1" ]]; then
  gcloud compute instances delete "$INSTANCE_NAME" --zone "$ZONE" --quiet || sudo shutdown -h now
else
  sudo shutdown -h now
fi
'@

Write-Utf8NoBom -Path $startupPath -Text $startup

$manifest = [ordered]@{
    project_id = $ProjectId
    bucket = $Bucket
    run_name = $RunName
    phase = $PhaseName
    teacher_module = $TeacherModule
    profile = $Profile
    opponent_profile = $OpponentProfile
    source_bucket = $SourceBucket
    seats = $Seats
    total_samples = $TotalSamples
    shard_samples = $ShardSamples
    record_skip_base = $RecordSkipBase
    record_skip_stride = $RecordSkipStride
    fast_skip_records = [bool]$FastSkipRecords
    total_shards = $totalShards
    vm_count = $VmCount
    machine_type = $MachineType
    cpu_threads = $CpuThreads
    zones = $Zones
    seed_stride = $SeedStride
    future_samples = $FutureSamples
    max_actions = $MaxActions
    opening_lookahead_samples = $OpeningLookaheadSamples
    candidate_model = $CandidateModel
    candidate_models = $CandidateModels
    candidate_topk = $CandidateTopk
    candidate_union_cap = $CandidateUnionCap
    candidate_union_mode = $CandidateUnionMode
    collect_topk_log = [bool]$CollectTopkLog
    hu_turn2_stage8b_model = $HuTurn2Stage8bModel
    source_uri = $sourceUri
    startup_uri = $startupUri
    self_delete = -not $NoSelfDelete
    production_default = "No-Go"
    large_teacher_status = $largeTeacherStatus
    additional_package_files = $additionalPackageEntries
    created_at = (Get-Date).ToUniversalTime().ToString("o")
}
Write-Utf8NoBom -Path $manifestPath -Text (($manifest | ConvertTo-Json -Depth 8) + "`n")

gcloud storage cp $packagePath $sourceUri --project $ProjectId | Out-Null
gcloud storage cp $startupPath $startupUri --project $ProjectId | Out-Null
gcloud storage cp $manifestPath $manifestUri --project $ProjectId | Out-Null
gcloud storage cp $shardManifestPath "gs://$Bucket/runs/$RunName/source/shards_manifest.jsonl" --project $ProjectId | Out-Null

$vmPrefix = Convert-ToVmName $RunName
$instances = @()
$workerIndices = if ($StartShardIndices.Count -gt 0) { $StartShardIndices } else { @(0..($VmCount - 1)) }
foreach ($i in $workerIndices) {
    $zone = $Zones[$i % $Zones.Count]
    $vmName = ("{0}-{1:d3}" -f $vmPrefix, $i)
    if ($vmName.Length -gt 63) { $vmName = $vmName.Substring(0, 63).Trim("-") }
    $metadata = @(
        "RUN_NAME=$RunName",
        "BUCKET=$Bucket",
        "TOTAL_SHARDS=$totalShards",
        "VM_COUNT=$VmCount",
        "START_SHARD=$i",
        "SOURCE_URI=$sourceUri",
        "FUTURE_SAMPLES=$FutureSamples",
        "MAX_ACTIONS=$MaxActions",
        "OPENING_LOOKAHEAD_SAMPLES=$OpeningLookaheadSamples",
        "PROFILE=$Profile",
        "OPPONENT_PROFILE=$OpponentProfile",
        "SOURCE_BUCKET=$SourceBucket",
        ("SEATS=" + ($Seats -join ";")),
        ("FAST_SKIP_RECORDS=" + ($(if ($FastSkipRecords) { "1" } else { "0" }))),
        "CANDIDATE_MODEL=$CandidateModel",
        ("CANDIDATE_MODELS=" + ($CandidateModels -join ";")),
        "CANDIDATE_TOPK=$CandidateTopk",
        "CANDIDATE_UNION_CAP=$CandidateUnionCap",
        "CANDIDATE_UNION_MODE=$CandidateUnionMode",
        "TEACHER_MODULE=$TeacherModule",
        "PHASE_NAME=$PhaseName",
        "CPU_THREADS=$CpuThreads",
        ("COLLECT_TOPK_LOG=" + ($(if ($CollectTopkLog) { "1" } else { "0" }))),
        ("SELF_DELETE=" + ($(if ($NoSelfDelete) { "0" } else { "1" })))
    ) -join ","

    $instances += [pscustomobject]@{ name = $vmName; zone = $zone; start_shard = $i }
    if ($CreateInstances) {
        if ($SkipExistingInstances) {
            $existing = gcloud compute instances list `
                --project $ProjectId `
                --filter ("name='" + $vmName + "'") `
                --format "value(name)" 2>$null
            if ($existing) { continue }
        }
        $createArgs = @(
            "compute", "instances", "create", $vmName,
            "--project", $ProjectId,
            "--zone", $zone,
            "--machine-type", $MachineType,
            "--image-family", "ubuntu-2404-lts-amd64",
            "--image-project", "ubuntu-os-cloud",
            "--boot-disk-size", ("{0}GB" -f $BootDiskGb),
            "--scopes", "https://www.googleapis.com/auth/cloud-platform",
            "--metadata", $metadata,
            "--metadata-from-file", "startup-script=$startupPath"
        )
        if (-not $NoSpot) {
            $createArgs += @(
                "--provisioning-model", "SPOT",
                "--instance-termination-action", "DELETE",
                "--maintenance-policy", "TERMINATE"
            )
        }
        & gcloud @createArgs | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Failed to create instance $vmName in $zone" }
    }
}

[pscustomobject]@{
    run_name = $RunName
    project_id = $ProjectId
    bucket = $Bucket
    phase = $PhaseName
    teacher_module = $TeacherModule
    profile = $Profile
    opponent_profile = $OpponentProfile
    seats = $Seats
    candidate_model = $CandidateModel
    candidate_models = $CandidateModels
    candidate_topk = $CandidateTopk
    candidate_union_cap = $CandidateUnionCap
    candidate_union_mode = $CandidateUnionMode
    total_samples = $TotalSamples
    shard_samples = $ShardSamples
    record_skip_base = $RecordSkipBase
    record_skip_stride = $RecordSkipStride
    fast_skip_records = [bool]$FastSkipRecords
    total_shards = $totalShards
    vm_count = $VmCount
    machine_type = $MachineType
    cpu_threads = $CpuThreads
    spot = -not $NoSpot
    package = $packagePath
    source_uri = $sourceUri
    manifest_uri = $manifestUri
    shard_manifest = $shardManifestPath
    additional_package_files = $additionalPackageEntries
    create_instances = [bool]$CreateInstances
    instances = $instances
} | ConvertTo-Json -Depth 6
