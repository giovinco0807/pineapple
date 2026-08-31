# ============================================================================
# Restart T3 data generation on completed VMs (PowerShell)
# ============================================================================
# Dispatches new chunk generation (chunks 3-5) with different seeds
# on all VMs that have completed their initial 3 chunks.
# ============================================================================

$PROJECT = "ofc-solver-485418"
$GCS_BASE = "gs://ofc-solver-results/t3_rust_fleet/run_20260509"
$STATES_PER_CHUNK = 1000
$SEED_OFFSET = 52000
$BTN_SAMPLES = 100

# Zone mapping
function Get-Zone($idx) {
    if ($idx -lt 24) { "us-east1-c" }
    elseif ($idx -lt 48) { "us-central1-a" }
    elseif ($idx -lt 72) { "us-east4-a" }
    elseif ($idx -lt 96) { "us-west1-a" }
    elseif ($idx -lt 122) { "asia-northeast1-b" }
    else { "asia-east1-a" }
}

# DONE + RUNNING VMs (excludes TERMINATED: 24,47,75,76,79,82,84,86,89,93)
$doneRunning = @(
    8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,
    25,26,27,28,29,30,31,34,37,38,39,40,41,46,
    48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,
    72,73,78,80,81,83,85,90,95,
    96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,
    122,123,124,125,126,127
)

$total = $doneRunning.Count
$newStates = $total * 3 * $STATES_PER_CHUNK
Write-Host "=== Restarting T3 generation on $total completed VMs ==="
Write-Host "    New chunks: 3-5 ($STATES_PER_CHUNK states each)"
Write-Host "    Seed offset: $SEED_OFFSET"
Write-Host "    Expected: $newStates new states"
Write-Host ""

# Generate restart command template
$restartCmd = @'
cd ~/ofc-pineapple
source "$HOME/.cargo/env" 2>/dev/null || true
VM_IDX=__VM_IDX__
RESULT_DIR="$HOME/ofc-pineapple/ai/data/t3_fleet/vm${VM_IDX}"
GCS_BASE="__GCS_BASE__"
for PART in 3 4 5; do
    CHUNK_DIR="$RESULT_DIR/chunk_${PART}"
    mkdir -p "$CHUNK_DIR"
    JSONL="$CHUNK_DIR/t3_vm${VM_IDX}_chunk${PART}.jsonl"
    SEED=$(( __SEED_OFFSET__ + ${VM_IDX} * 1000 + ${PART} ))
    ./ai/rust_solver/target/release/t3_generator \
        --states __STATES__ \
        --mode bb \
        --btn-samples __BTN__ \
        --seed $SEED \
        --onnx-model ai/data/t4_oracle_v2/t4_oracle.onnx \
        --output $JSONL \
        --log-interval 100 2>&1 | tee "$CHUNK_DIR/generate.log"
    python3 ai/training/convert_t3_rust_jsonl.py \
        --input $JSONL \
        --output-dir $CHUNK_DIR \
        --chunk-size __STATES__ 2>&1 | tee "$CHUNK_DIR/convert.log"
    gsutil -m cp "$CHUNK_DIR"/*.npz "$CHUNK_DIR"/*.log \
        ${GCS_BASE}/vm${VM_IDX}/chunk_${PART}/ 2>/dev/null || true
done
echo "DONE_R2 vm=${VM_IDX} states=$(( __STATES__ * 3 ))" | tee "$RESULT_DIR/DONE_R2"
gsutil cp "$RESULT_DIR/DONE_R2" ${GCS_BASE}/vm${VM_IDX}/DONE_R2
'@

# Dispatch in batches
$BATCH_SIZE = 15
$batchNum = 0

for ($i = 0; $i -lt $total; $i += $BATCH_SIZE) {
    $batchNum++
    $batchEnd = [Math]::Min($i + $BATCH_SIZE, $total)
    Write-Host "--- Batch $batchNum ($($batchEnd - $i) VMs) ---"

    $jobs = @()
    for ($j = $i; $j -lt $batchEnd; $j++) {
        $vmIdx = $doneRunning[$j]
        $zone = Get-Zone $vmIdx
        $instance = "ofc-t3fleet-$vmIdx"

        # Build per-VM command
        $cmd = $restartCmd -replace '__VM_IDX__', $vmIdx `
                           -replace '__GCS_BASE__', $GCS_BASE `
                           -replace '__SEED_OFFSET__', $SEED_OFFSET `
                           -replace '__STATES__', $STATES_PER_CHUNK `
                           -replace '__BTN__', $BTN_SAMPLES

        $jobs += Start-Job -ScriptBlock {
            param($inst, $z, $proj, $remoteCmd)
            $result = gcloud compute ssh $inst --zone=$z --project=$proj --command="nohup bash -c '$remoteCmd' > /tmp/restart_r2.log 2>&1 &; echo DISPATCHED" 2>&1
            "$inst ($z): $($result | Select-Object -Last 1)"
        } -ArgumentList $instance, $zone, $PROJECT, $cmd

        Write-Host "  Dispatching $instance ($zone)..."
    }

    # Wait for batch
    $results = $jobs | Wait-Job -Timeout 120 | Receive-Job
    $jobs | Remove-Job -Force -ErrorAction SilentlyContinue
    foreach ($r in $results) {
        Write-Host "  $r"
    }
    Write-Host ""
}

Write-Host "=== All $total VMs dispatched ==="
Write-Host "Expected additional: $newStates states (~264,000)"
Write-Host ""
Write-Host "Monitor progress:"
Write-Host "  gsutil ls gs://ofc-solver-results/t3_rust_fleet/run_20260509/*/DONE_R2 | Measure-Object"
