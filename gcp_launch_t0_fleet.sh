#!/bin/bash
# ============================================================================
# GCP T0 Data Generation Fleet Launcher (Phase B)
# ============================================================================
# Creates multiple e2-highcpu-16 VMs to generate T0 training data in parallel.
# Each VM clones the repo, builds the Rust CFR solver, generates data,
# uploads to GCS, and self-deletes.
#
# Target: 5,000 hands across 10 workers (500 hands each)
# Nesting: [3,2,1] (6 sims/action)
# Samples: 50 per placement
# Estimated time: ~2 days per worker
# Estimated cost: ~$30-40 total
#
# Usage:
#   bash gcp_launch_t0_fleet.sh           # Launch all workers
#   bash gcp_launch_t0_fleet.sh --dry-run # Show commands without executing
#   bash gcp_launch_t0_fleet.sh --workers 5 --hands 200  # Custom config
# ============================================================================

set -e

PROJECT="ofc-solver-485418"
STARTUP_SCRIPT="gcp_worker_startup_v2.sh"

# Default configuration
N_WORKERS=${WORKERS:-10}
HANDS_PER_WORKER=${HANDS:-500}
SAMPLES=${SAMPLES:-50}
NESTING=${NESTING:-"3,2,1"}
BASE_SEED=${BASE_SEED:-500000}
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --workers) N_WORKERS="$2"; shift 2 ;;
        --hands) HANDS_PER_WORKER="$2"; shift 2 ;;
        --samples) SAMPLES="$2"; shift 2 ;;
        --nesting) NESTING="$2"; shift 2 ;;
        --seed) BASE_SEED="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

TOTAL_HANDS=$((N_WORKERS * HANDS_PER_WORKER))

# Zone pool - distribute across regions for availability
ZONES=(
    "us-central1-a"
    "us-central1-b"
    "us-central1-c"
    "us-east1-b"
    "us-east1-c"
    "us-east4-a"
    "us-east4-c"
    "us-west1-a"
    "us-west1-b"
    "us-west4-a"
    "europe-west1-b"
    "europe-west1-c"
    "europe-west4-a"
    "northamerica-northeast1-a"
    "us-south1-a"
    "us-west2-a"
)

echo "============================================"
echo "  T0 Data Generation Fleet Launcher"
echo "============================================"
echo "  Project:     $PROJECT"
echo "  Workers:     $N_WORKERS"
echo "  Hands/worker: $HANDS_PER_WORKER"
echo "  Total hands: $TOTAL_HANDS"
echo "  Samples:     $SAMPLES"
echo "  Nesting:     $NESTING"
echo "  Base seed:   $BASE_SEED"
echo "  Startup:     $STARTUP_SCRIPT"
echo "  Dry run:     $DRY_RUN"
echo ""

# Verify startup script exists
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ ! -f "$SCRIPT_DIR/$STARTUP_SCRIPT" ]; then
    echo "ERROR: $STARTUP_SCRIPT not found in $SCRIPT_DIR"
    exit 1
fi

# Estimate cost
# e2-highcpu-16: ~$0.38/hr
SECS_PER_HAND=300  # conservative estimate
EST_HOURS=$(echo "scale=1; $HANDS_PER_WORKER * $SECS_PER_HAND / 3600" | bc)
COST_PER_VM=$(echo "scale=2; $EST_HOURS * 0.38" | bc)
TOTAL_COST=$(echo "scale=2; $COST_PER_VM * $N_WORKERS" | bc)

echo "  Estimated time/worker: ~${EST_HOURS}h"
echo "  Estimated cost/worker: ~\$${COST_PER_VM}"
echo "  Estimated total cost:  ~\$${TOTAL_COST}"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "--- DRY RUN: Commands that would be executed ---"
    echo ""
fi

# Launch workers
LAUNCHED=0
FAILED=0

for i in $(seq 0 $((N_WORKERS - 1))); do
    WORKER_NAME="t0-gen-${i}"
    ZONE_IDX=$((i % ${#ZONES[@]}))
    ZONE="${ZONES[$ZONE_IDX]}"
    SEED=$((BASE_SEED + i * HANDS_PER_WORKER * 100))

    echo "  [$((i+1))/$N_WORKERS] Creating $WORKER_NAME in $ZONE (seed=$SEED)..."

    CMD="gcloud compute instances create $WORKER_NAME \
        --project=$PROJECT \
        --zone=$ZONE \
        --machine-type=e2-highcpu-16 \
        --image-family=ubuntu-2204-lts \
        --image-project=ubuntu-os-cloud \
        --boot-disk-size=30GB \
        --boot-disk-type=pd-ssd \
        --scopes=storage-rw,logging-write,monitoring-write \
        --metadata=worker-id=$i,seed=$SEED,hands=$HANDS_PER_WORKER,samples=$SAMPLES,nesting=$NESTING \
        --metadata-from-file=startup-script=$SCRIPT_DIR/$STARTUP_SCRIPT \
        --no-restart-on-failure"

    if [ "$DRY_RUN" = true ]; then
        echo "    $CMD"
        echo ""
    else
        if eval "$CMD" 2>&1 | tail -1; then
            LAUNCHED=$((LAUNCHED + 1))
            echo "    ✅ Created"
        else
            FAILED=$((FAILED + 1))
            echo "    ❌ Failed (zone $ZONE may be out of capacity)"
        fi
    fi
done

echo ""
echo "============================================"
echo "  Fleet Launch Summary"
echo "============================================"
if [ "$DRY_RUN" = true ]; then
    echo "  [DRY RUN] $N_WORKERS workers would be created"
else
    echo "  Launched: $LAUNCHED / $N_WORKERS"
    echo "  Failed:   $FAILED"
fi
echo ""
echo "  GCS output: gs://ofc-solver-485418/t0_phase_b/"
echo ""
echo "  Monitor progress:"
echo "    gsutil ls -l gs://ofc-solver-485418/t0_phase_b/"
echo ""
echo "  Count completed hands:"
echo "    python -c \"import subprocess; r=subprocess.run(['gsutil','cat','gs://ofc-solver-485418/t0_phase_b/worker_*'],capture_output=True,text=True); print(f'{len(r.stdout.strip().splitlines())} hands completed')\""
echo ""
echo "  Check worker status:"
echo "    gcloud compute instances list --filter='name:t0-gen-'"
echo ""
echo "  SSH into a worker:"
echo "    gcloud compute ssh t0-gen-0 --zone=us-central1-a"
echo ""
echo "  View logs:"
echo "    gcloud compute ssh t0-gen-0 --zone=us-central1-a --command='tail -f /tmp/worker_0.log'"
echo ""
echo "  Download all data:"
echo "    gsutil -m cp 'gs://ofc-solver-485418/t0_phase_b/worker_*.jsonl' ai/data/t0_phase_b/"
echo ""
echo "  VMs will self-delete after completion."
echo "============================================"
