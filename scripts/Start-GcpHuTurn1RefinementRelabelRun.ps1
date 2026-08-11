param(
    [string]$ProjectId = "ofc-solver-485418",
    [string]$Bucket = "pokerhu-ofc-solver-485418-training",
    [string]$RunName = ("regular-hu-t1-refinement-relabel-" + (Get-Date -Format "yyyyMMdd-HHmmss")),
    [string]$TargetInput = "outputs/hu_turn1_stage1j_stratified_teacher/gcp_natural_mc32_partial110/refinement_targets/targets.jsonl",
    [int]$TotalTargets = 100,
    [int]$ShardTargets = 2,
    [int]$BaseSeed = 2026062505,
    [int]$SeedStride = 1000000,
    [int]$VmCount = 50,
    [string]$MachineType = "e2-highcpu-4",
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
    [int]$FutureSamples = 64,
    [int]$MaxActions = 0,
    [string]$CandidateModel = "",
    [string[]]$CandidateModels = @(),
    [int]$CandidateTopk = 0,
    [int]$CandidateUnionCap = 0,
    [string]$CandidateUnionMode = "min_rank",
    [switch]$SourceActionsOnly,
    [int]$OpeningLookaheadSamples = 1,
    [double]$TargetTimeoutSeconds = 0,
    [int]$ShardTimeoutSeconds = 0,
    [switch]$ContinueOnTargetError,
    [string]$Profile = "stage9f_p2",
    [string]$OpponentProfile = "stage9f_p2",
    [string]$HuTurn2Stage8bModel = "models/hu_turn2_stage8b_safe_lcb196_20k_reference_override_cached_rank_wide.pt",
    [int]$BootDiskGb = 50,
    [string[]]$StartShards = @(),
    [switch]$CreateInstances,
    [switch]$SkipExistingInstances,
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

if ($TotalTargets -le 0) { throw "TotalTargets must be positive" }
if ($ShardTargets -le 0) { throw "ShardTargets must be positive" }
if ($VmCount -le 0) { throw "VmCount must be positive" }
if ($FutureSamples -le 0) { throw "FutureSamples must be positive" }
if ($MaxActions -lt 0) { throw "MaxActions must be non-negative" }
if ($CandidateTopk -lt 0) { throw "CandidateTopk must be non-negative" }
if ($CandidateUnionCap -lt 0) { throw "CandidateUnionCap must be non-negative" }
if ($CandidateUnionMode -notin @("min_rank", "rank_sum", "reciprocal_rank_sum", "mean_score", "max_score", "mean_z_score", "max_z_score")) {
    throw "CandidateUnionMode is invalid: $CandidateUnionMode"
}
if ($CandidateModel -and $CandidateModels.Count -gt 0) { throw "CandidateModel and CandidateModels are mutually exclusive" }
if ($CandidateModel -and $CandidateTopk -le 0) { throw "CandidateTopk must be positive when CandidateModel is set" }
if ($CandidateModels.Count -gt 0 -and $CandidateTopk -le 0) { throw "CandidateTopk must be positive when CandidateModels is set" }
if ($OpeningLookaheadSamples -lt 0) { throw "OpeningLookaheadSamples must be non-negative" }
if ($TargetTimeoutSeconds -lt 0) { throw "TargetTimeoutSeconds must be non-negative" }
if ($ShardTimeoutSeconds -lt 0) { throw "ShardTimeoutSeconds must be non-negative" }
if (-not $Zones -or $Zones.Count -eq 0) { throw "At least one zone is required" }
if ($RunName -notmatch '^[a-zA-Z0-9][a-zA-Z0-9._-]*$') {
    throw "RunName may only contain letters, numbers, dot, underscore, and dash"
}
if ([System.IO.Path]::IsPathRooted($TargetInput)) {
    throw "TargetInput must be repo-relative"
}
if ($CandidateModel -and [System.IO.Path]::IsPathRooted($CandidateModel)) {
    throw "CandidateModel must be repo-relative"
}
foreach ($candidatePath in $CandidateModels) {
    if ([System.IO.Path]::IsPathRooted($candidatePath)) {
        throw "CandidateModels entries must be repo-relative: $candidatePath"
    }
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
$targetPath = Join-Path $repoRoot $TargetInput
if (-not (Test-Path $targetPath)) { throw "TargetInput not found: $TargetInput" }

$requiredModels = @(
    "models/opening_stage7_torch_wide.pt",
    "models/turn1_stage6_torch_wide.pt",
    "models/turn2_stage8.pkl",
    "models/turn3_stage6.pkl",
    "models/hu_turn3_stage7_reference_override_cached_rank_wide.pt",
    "models/hu_turn3_stage3_mc32_500k_plus_m8_12_f128_100k_w2_cached_rank_wide.pt",
    $HuTurn2Stage8bModel
) + @($CandidateModel) + @($CandidateModels) | Where-Object { $_ } | Select-Object -Unique
foreach ($model in $requiredModels) {
    if ([System.IO.Path]::IsPathRooted($model)) { throw "Model paths must be repo-relative: $model" }
    if (-not (Test-Path (Join-Path $repoRoot $model))) { throw "Required model not found: $model" }
}

$totalShards = [int][Math]::Ceiling($TotalTargets / [double]$ShardTargets)
$startShardIndices = @()
foreach ($startShardValue in $StartShards) {
    foreach ($part in ($startShardValue -split ",")) {
        $trimmed = $part.Trim()
        if ($trimmed) { $startShardIndices += [int]$trimmed }
    }
}
$startShardIndices = @($startShardIndices | Sort-Object -Unique)
foreach ($startShard in $startShardIndices) {
    if ($startShard -lt 0 -or $startShard -ge $totalShards) {
        throw "StartShards values must be in [0, total_shards): $startShard"
    }
}

$shardSpecs = New-Object System.Collections.Generic.List[object]
for ($shardIndex = 0; $shardIndex -lt $totalShards; $shardIndex += 1) {
    $offset = $shardIndex * $ShardTargets
    $count = [Math]::Min($ShardTargets, $TotalTargets - $offset)
    $seed = $BaseSeed + ($shardIndex * $SeedStride)
    $shardSpecs.Add([ordered]@{
        shard = $shardIndex
        offset = $offset
        count = $count
        seed = $seed
        output_prefix = ("shard_{0:D3}_seed{1}_targets{2:D05}" -f $shardIndex, $seed, $count)
    })
}

if ($DryRun) {
    [pscustomobject]@{
        execution = "dry_run"
        run_name = $RunName
        project_id = $ProjectId
        bucket = $Bucket
        phase = "hu_t1_refinement_relabel"
        target_input = $TargetInput
        total_targets = $TotalTargets
        shard_targets = $ShardTargets
        total_shards = $totalShards
        vm_count = $VmCount
        machine_type = $MachineType
        future_samples = $FutureSamples
        max_actions = $MaxActions
        candidate_model = $CandidateModel
        candidate_models = $CandidateModels
        candidate_topk = $CandidateTopk
        candidate_union_cap = $CandidateUnionCap
        candidate_union_mode = $CandidateUnionMode
        source_actions_only = [bool]$SourceActionsOnly
        target_timeout_seconds = $TargetTimeoutSeconds
        shard_timeout_seconds = $ShardTimeoutSeconds
        continue_on_target_error = [bool]$ContinueOnTargetError
        profile = $Profile
        opponent_profile = $OpponentProfile
        create_instances = [bool]$CreateInstances
        production_default = "No-Go"
        t1_runtime = "No-Go"
        shards = $shardSpecs
    } | ConvertTo-Json -Depth 8
    exit 0
}

$runDir = Join-Path $repoRoot ("outputs/gcp_runs/{0}" -f $RunName)
$packageDir = Join-Path $runDir "package_src"
$packagePath = Join-Path $runDir "ofc_regular_hu_t1_refinement_relabel_source.zip"
$startupPath = Join-Path $runDir "startup_hu_t1_refinement_relabel_spot.sh"
$manifestPath = Join-Path $runDir "manifest.json"
$shardManifestPath = Join-Path $runDir "shards_manifest.jsonl"
$sourceUri = "gs://$Bucket/runs/$RunName/source/ofc_regular_hu_t1_refinement_relabel_source.zip"
$startupUri = "gs://$Bucket/runs/$RunName/source/startup_hu_t1_refinement_relabel_spot.sh"
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
$targetDest = Join-Path $packageDir $TargetInput
New-Item -ItemType Directory -Force -Path (Split-Path $targetDest -Parent) | Out-Null
Copy-Item -LiteralPath $targetPath -Destination $targetDest

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
TARGET_INPUT="$(meta TARGET_INPUT)"
FUTURE_SAMPLES="$(meta FUTURE_SAMPLES)"
MAX_ACTIONS="$(meta MAX_ACTIONS)"
CANDIDATE_MODEL="$(meta CANDIDATE_MODEL)"
CANDIDATE_MODELS="$(meta CANDIDATE_MODELS)"
CANDIDATE_TOPK="$(meta CANDIDATE_TOPK)"
CANDIDATE_UNION_CAP="$(meta CANDIDATE_UNION_CAP)"
CANDIDATE_UNION_MODE="$(meta CANDIDATE_UNION_MODE)"
SOURCE_ACTIONS_ONLY="$(meta SOURCE_ACTIONS_ONLY)"
OPENING_LOOKAHEAD_SAMPLES="$(meta OPENING_LOOKAHEAD_SAMPLES)"
TARGET_TIMEOUT_SECONDS="$(meta TARGET_TIMEOUT_SECONDS)"
SHARD_TIMEOUT_SECONDS="$(meta SHARD_TIMEOUT_SECONDS)"
CONTINUE_ON_TARGET_ERROR="$(meta CONTINUE_ON_TARGET_ERROR)"
PROFILE="$(meta PROFILE)"
OPPONENT_PROFILE="$(meta OPPONENT_PROFILE)"
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
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

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

WORK_DIR="/opt/ofc-regular-hu-t1-refinement-relabel"
sudo rm -rf "$WORK_DIR"
sudo mkdir -p "$WORK_DIR"
sudo chown "$CURRENT_UID:$CURRENT_GID" "$WORK_DIR"
gcloud storage cp "$SOURCE_URI" /tmp/ofc_regular_hu_t1_refinement_relabel_source.zip
unzip -q /tmp/ofc_regular_hu_t1_refinement_relabel_source.zip -d "$WORK_DIR"
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

STATUS_DIR="/tmp/ofc-hu-t1-refinement-relabel-status"
RESULT_DIR="/tmp/ofc-hu-t1-refinement-relabel-results"
mkdir -p "$STATUS_DIR" "$RESULT_DIR"

log "start HU T1 refinement relabel loop run=$RUN_NAME start=$START_SHARD stride=$VM_COUNT total_shards=$TOTAL_SHARDS"
for (( shard=START_SHARD; shard<TOTAL_SHARDS; shard+=VM_COUNT )); do
  spec="$(sed -n "$((shard + 1))p" shards_manifest.jsonl)"
  if [[ -z "$spec" ]]; then
    log "missing spec for shard=$shard"
    exit 2
  fi
  offset="$(json_field "$spec" offset)"
  count="$(json_field "$spec" count)"
  seed="$(json_field "$spec" seed)"
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
  log "run shard=$shard offset=$offset count=$count seed=$seed profile=$PROFILE opponent=$OPPONENT_PROFILE"

  cmd=(
    python -B -m ofc_regular.relabel_hu_turn1_refinement_targets
    --input "$TARGET_INPUT"
    --output "${local_out}/relabel.jsonl"
    --summary-output "${local_out}/summary.json"
    --skip-output "${local_out}/skipped.jsonl"
    --profile "$PROFILE"
    --opponent-profile "$OPPONENT_PROFILE"
    --future-samples "$FUTURE_SAMPLES"
    --max-actions "$MAX_ACTIONS"
    --skip-targets "$offset"
    --max-targets "$count"
    --seed "$seed"
    --opening-lookahead-samples "$OPENING_LOOKAHEAD_SAMPLES"
    --target-timeout-seconds "$TARGET_TIMEOUT_SECONDS"
  )
  if [[ -n "${CANDIDATE_MODEL:-}" ]]; then
    cmd+=(--candidate-model "$CANDIDATE_MODEL" --candidate-topk "$CANDIDATE_TOPK")
  fi
  if [[ -n "${CANDIDATE_MODELS:-}" ]]; then
    IFS=';' read -r -a candidate_model_paths <<< "$CANDIDATE_MODELS"
    cmd+=(--candidate-models)
    for candidate_model_path in "${candidate_model_paths[@]}"; do
      if [[ -n "$candidate_model_path" ]]; then
        cmd+=("$candidate_model_path")
      fi
    done
    cmd+=(--candidate-topk "$CANDIDATE_TOPK")
  fi
  if [[ "${CANDIDATE_UNION_CAP:-0}" -gt 0 ]]; then
    cmd+=(--candidate-union-cap "$CANDIDATE_UNION_CAP")
  fi
  if [[ -n "${CANDIDATE_UNION_MODE:-}" ]]; then
    cmd+=(--candidate-union-mode "$CANDIDATE_UNION_MODE")
  fi
  if [[ "$SOURCE_ACTIONS_ONLY" == "1" ]]; then
    cmd+=(--source-actions-only)
  fi
  if [[ "$CONTINUE_ON_TARGET_ERROR" == "1" ]]; then
    cmd+=(--continue-on-target-error)
  fi

  set +e
  if [[ "${SHARD_TIMEOUT_SECONDS:-0}" -gt 0 ]]; then
    timeout --kill-after=60 "${SHARD_TIMEOUT_SECONDS}s" "${cmd[@]}" > "$log_path" 2>&1
  else
    "${cmd[@]}" > "$log_path" 2>&1
  fi
  exit_code=$?
  set -e

  finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  end_seconds="$(date +%s)"
  elapsed=$((end_seconds - start_seconds))
  if [[ "$exit_code" -eq 0 ]]; then
    printf 'complete\n' > "${local_out}/DONE"
    gcloud storage cp "$local_out"/* "gs://${BUCKET}/runs/${RUN_NAME}/results/${output_prefix}/" >/dev/null
    cat > "$status_path" <<EOF
{"run_name":"$RUN_NAME","phase":"hu_t1_refinement_relabel","shard":$shard,"offset":$offset,"count":$count,"seed":$seed,"profile":"$PROFILE","opponent_profile":"$OPPONENT_PROFILE","future_samples":$FUTURE_SAMPLES,"target_timeout_seconds":$TARGET_TIMEOUT_SECONDS,"shard_timeout_seconds":$SHARD_TIMEOUT_SECONDS,"continue_on_target_error":"$CONTINUE_ON_TARGET_ERROR","status":"complete","instance":"$INSTANCE_NAME","zone":"$ZONE","elapsed_seconds":$elapsed,"started_at":"$started_at","finished_at":"$finished_at","output":"gs://${BUCKET}/runs/${RUN_NAME}/results/${output_prefix}"}
EOF
    gcloud storage cp "$status_path" "$status_remote" >/dev/null || true
  elif [[ "$exit_code" -eq 124 ]]; then
    python - "$TARGET_INPUT" "$offset" "$count" "$PROFILE" "$OPPONENT_PROFILE" "$FUTURE_SAMPLES" "$MAX_ACTIONS" "$TARGET_TIMEOUT_SECONDS" "$SHARD_TIMEOUT_SECONDS" "${local_out}/relabel.jsonl" "${local_out}/skipped.jsonl" "${local_out}/summary.json" <<'PY'
import json
import sys
from pathlib import Path

target_input = Path(sys.argv[1])
offset = int(sys.argv[2])
count = int(sys.argv[3])
profile = sys.argv[4]
opponent_profile = sys.argv[5]
future_samples = int(sys.argv[6])
max_actions = int(sys.argv[7])
target_timeout_seconds = float(sys.argv[8])
shard_timeout_seconds = int(sys.argv[9])
output = Path(sys.argv[10])
skipped_output = Path(sys.argv[11])
summary_output = Path(sys.argv[12])

targets = []
with target_input.open("r", encoding="utf-8-sig") as handle:
    for index, line in enumerate(handle):
        if not line.strip():
            continue
        if index < offset:
            continue
        if len(targets) >= count:
            break
        targets.append(json.loads(line))

skipped = []
for record in targets:
    skipped.append({
        "schema": "hu_turn1_stage1_refinement_relabel_skip_v1",
        "target_id": record.get("target_id", record.get("sample_id")),
        "sample_id": record.get("sample_id"),
        "hand_seed": record.get("hand_seed"),
        "player": record.get("player"),
        "seat": record.get("seat"),
        "source_bucket": record.get("source_bucket"),
        "source_bucket_group": record.get("source_bucket_group"),
        "selection_reasons": list(record.get("selection_reasons") or []),
        "skip_reason": "shard_timeout",
        "skip_message": f"shard timeout killed relabel process after {shard_timeout_seconds}s",
        "profile": profile,
        "opponent_profile": opponent_profile,
        "future_samples": future_samples,
        "max_actions": max_actions,
        "target_timeout_seconds": target_timeout_seconds,
        "shard_timeout_seconds": shard_timeout_seconds,
    })

output.write_text("", encoding="utf-8")
with skipped_output.open("w", encoding="utf-8") as handle:
    for row in skipped:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
summary = {
    "schema": "hu_turn1_stage1_refinement_relabel_summary_v1",
    "input": str(target_input),
    "output": str(output),
    "skip_output": str(skipped_output),
    "target_attempts": len(targets),
    "records": 0,
    "skipped_targets": len(skipped),
    "timed_out_targets": len(skipped),
    "failed_targets": 0,
    "skip_reason_counts": {"shard_timeout": len(skipped)},
    "profile": profile,
    "opponent_profile": opponent_profile,
    "future_samples": future_samples,
    "max_actions": max_actions,
    "target_timeout_seconds": target_timeout_seconds,
    "target_timeout_supported": True,
    "continue_on_target_error": True,
    "best_action_changed": 0,
    "best_action_changed_rate": 0.0,
    "source_best_new_regret_mean": 0.0,
    "source_best_new_regret_max": 0.0,
    "seat_counts": {},
    "reason_counts": {},
    "relabel_seconds_sum": 0.0,
    "t2_choose_action_seconds_sum": 0.0,
}
summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
    printf 'complete_with_shard_timeout_skip\n' > "${local_out}/DONE"
    gcloud storage cp "$local_out"/* "gs://${BUCKET}/runs/${RUN_NAME}/results/${output_prefix}/" >/dev/null
    gcloud storage cp "$log_path" "gs://${BUCKET}/runs/${RUN_NAME}/logs/${output_prefix}.log" >/dev/null || true
    cat > "$status_path" <<EOF
{"run_name":"$RUN_NAME","phase":"hu_t1_refinement_relabel","shard":$shard,"offset":$offset,"count":$count,"seed":$seed,"profile":"$PROFILE","opponent_profile":"$OPPONENT_PROFILE","future_samples":$FUTURE_SAMPLES,"target_timeout_seconds":$TARGET_TIMEOUT_SECONDS,"shard_timeout_seconds":$SHARD_TIMEOUT_SECONDS,"continue_on_target_error":"$CONTINUE_ON_TARGET_ERROR","status":"complete_with_shard_timeout_skip","instance":"$INSTANCE_NAME","zone":"$ZONE","exit_code":$exit_code,"elapsed_seconds":$elapsed,"started_at":"$started_at","finished_at":"$finished_at","output":"gs://${BUCKET}/runs/${RUN_NAME}/results/${output_prefix}","log":"gs://${BUCKET}/runs/${RUN_NAME}/logs/${output_prefix}.log"}
EOF
    gcloud storage cp "$status_path" "$status_remote" >/dev/null || true
  else
    gcloud storage cp "$log_path" "gs://${BUCKET}/runs/${RUN_NAME}/logs/${output_prefix}.log" >/dev/null || true
    cat > "$status_path" <<EOF
{"run_name":"$RUN_NAME","phase":"hu_t1_refinement_relabel","shard":$shard,"offset":$offset,"count":$count,"seed":$seed,"profile":"$PROFILE","opponent_profile":"$OPPONENT_PROFILE","future_samples":$FUTURE_SAMPLES,"target_timeout_seconds":$TARGET_TIMEOUT_SECONDS,"shard_timeout_seconds":$SHARD_TIMEOUT_SECONDS,"continue_on_target_error":"$CONTINUE_ON_TARGET_ERROR","status":"failed","instance":"$INSTANCE_NAME","zone":"$ZONE","exit_code":$exit_code,"elapsed_seconds":$elapsed,"started_at":"$started_at","finished_at":"$finished_at","log":"gs://${BUCKET}/runs/${RUN_NAME}/logs/${output_prefix}.log"}
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
    phase = "hu_t1_refinement_relabel"
    target_input = $TargetInput
    total_targets = $TotalTargets
    shard_targets = $ShardTargets
    total_shards = $totalShards
    vm_count = $VmCount
    machine_type = $MachineType
    zones = $Zones
    seed_stride = $SeedStride
    future_samples = $FutureSamples
    max_actions = $MaxActions
    candidate_model = $CandidateModel
    candidate_models = $CandidateModels
    candidate_topk = $CandidateTopk
    candidate_union_cap = $CandidateUnionCap
    candidate_union_mode = $CandidateUnionMode
    source_actions_only = [bool]$SourceActionsOnly
    target_timeout_seconds = $TargetTimeoutSeconds
    shard_timeout_seconds = $ShardTimeoutSeconds
    continue_on_target_error = [bool]$ContinueOnTargetError
    profile = $Profile
    opponent_profile = $OpponentProfile
    opening_lookahead_samples = $OpeningLookaheadSamples
    hu_turn2_stage8b_model = $HuTurn2Stage8bModel
    source_uri = $sourceUri
    startup_uri = $startupUri
    self_delete = -not $NoSelfDelete
    production_default = "No-Go"
    t1_runtime = "No-Go"
    created_at = (Get-Date).ToUniversalTime().ToString("o")
}
Write-Utf8NoBom -Path $manifestPath -Text (($manifest | ConvertTo-Json -Depth 8) + "`n")

gcloud storage cp $packagePath $sourceUri --project $ProjectId | Out-Null
gcloud storage cp $startupPath $startupUri --project $ProjectId | Out-Null
gcloud storage cp $manifestPath $manifestUri --project $ProjectId | Out-Null
gcloud storage cp $shardManifestPath "gs://$Bucket/runs/$RunName/source/shards_manifest.jsonl" --project $ProjectId | Out-Null

$vmPrefix = Convert-ToVmName $RunName
$instances = @()
$workerIndices = if ($startShardIndices.Count -gt 0) { $startShardIndices } else { @(0..($VmCount - 1)) }
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
        "TARGET_INPUT=$TargetInput",
        "FUTURE_SAMPLES=$FutureSamples",
        "MAX_ACTIONS=$MaxActions",
        "CANDIDATE_MODEL=$CandidateModel",
        ("CANDIDATE_MODELS=" + (($CandidateModels | Where-Object { $_ }) -join ";")),
        "CANDIDATE_TOPK=$CandidateTopk",
        "CANDIDATE_UNION_CAP=$CandidateUnionCap",
        "CANDIDATE_UNION_MODE=$CandidateUnionMode",
        ("SOURCE_ACTIONS_ONLY=" + ($(if ($SourceActionsOnly) { "1" } else { "0" }))),
        "OPENING_LOOKAHEAD_SAMPLES=$OpeningLookaheadSamples",
        "TARGET_TIMEOUT_SECONDS=$TargetTimeoutSeconds",
        "SHARD_TIMEOUT_SECONDS=$ShardTimeoutSeconds",
        ("CONTINUE_ON_TARGET_ERROR=" + ($(if ($ContinueOnTargetError) { "1" } else { "0" }))),
        "PROFILE=$Profile",
        "OPPONENT_PROFILE=$OpponentProfile",
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
        gcloud compute instances create $vmName `
            --project $ProjectId `
            --zone $zone `
            --machine-type $MachineType `
            --provisioning-model SPOT `
            --instance-termination-action DELETE `
            --maintenance-policy TERMINATE `
            --image-family ubuntu-2404-lts-amd64 `
            --image-project ubuntu-os-cloud `
            --boot-disk-size ("{0}GB" -f $BootDiskGb) `
            --scopes https://www.googleapis.com/auth/cloud-platform `
            --metadata $metadata `
            --metadata-from-file startup-script=$startupPath | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Failed to create instance $vmName in $zone" }
    }
}

[pscustomobject]@{
    run_name = $RunName
    project_id = $ProjectId
    bucket = $Bucket
    phase = "hu_t1_refinement_relabel"
    target_input = $TargetInput
    total_targets = $TotalTargets
    shard_targets = $ShardTargets
    total_shards = $totalShards
    vm_count = $VmCount
    machine_type = $MachineType
    future_samples = $FutureSamples
    max_actions = $MaxActions
    candidate_model = $CandidateModel
    candidate_models = $CandidateModels
    candidate_topk = $CandidateTopk
    candidate_union_cap = $CandidateUnionCap
    candidate_union_mode = $CandidateUnionMode
    source_actions_only = [bool]$SourceActionsOnly
    package = $packagePath
    source_uri = $sourceUri
    manifest_uri = $manifestUri
    shard_manifest = $shardManifestPath
    create_instances = [bool]$CreateInstances
    instances = $instances
} | ConvertTo-Json -Depth 6
