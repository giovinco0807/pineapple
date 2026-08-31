#!/bin/bash
# Transition script: download s50 data, then restart with s200/4800 hands
set -e

PROJECT="ofc-solver-485418"
ZONES=("us-central1-a" "us-central1-b" "us-central1-c" "us-central1-f" "us-east1-b" "us-east1-c" "us-east1-d" "us-east4-a" "us-east4-b" "us-east4-c")
PREFIX="ofc-teacher"
NUM_VMS=30
WORKERS_PER_VM=8

# NEW settings
NEW_TOTAL_HANDS=4800
NEW_SIMS=200
NEW_SEED=42
NEW_HANDS_PER_VM=$((NEW_TOTAL_HANDS / NUM_VMS))  # 160
NEW_HANDS_PER_WORKER=$((NEW_HANDS_PER_VM / WORKERS_PER_VM))  # 20

# OLD settings (for download)
OLD_SIMS=50

vm_zone() { echo "${ZONES[$(($1 % ${#ZONES[@]}))]}" ; }

# ---- Step 1: Download old s50 data ----
echo "=== Step 1: Downloading s50 data from all VMs ==="
LOCAL_DEST="/d/ofc_data/mc_teacher_s50_v2"
mkdir -p "$LOCAL_DEST"

for i in $(seq 0 $((NUM_VMS - 1))); do
    instance="${PREFIX}-${i}"
    z=$(vm_zone $i)
    (
        for w in $(seq 0 $((WORKERS_PER_VM - 1))); do
            sid=$((i * WORKERS_PER_VM + w))
            gcloud compute ssh "$instance" --zone="$z" --command="
                cat ~/ofc-pineapple/teacher_results/mc_s${OLD_SIMS}_shard${sid}.jsonl 2>/dev/null
            " 2>/dev/null > "$LOCAL_DEST/mc_s${OLD_SIMS}_shard${sid}.jsonl"
        done
        echo "  ${instance}: downloaded"
    ) &
done
wait
echo "=== Download complete ==="

# Merge
echo "=== Merging s50 data ==="
cat "$LOCAL_DEST"/mc_s${OLD_SIMS}_shard*.jsonl > "$LOCAL_DEST/mc_s${OLD_SIMS}_merged.jsonl"
total_lines=$(wc -l < "$LOCAL_DEST/mc_s${OLD_SIMS}_merged.jsonl")
echo "  Total records: ${total_lines}"

# ---- Step 2: Kill old processes and clear results ----
echo ""
echo "=== Step 2: Killing old processes and clearing results ==="
for i in $(seq 0 $((NUM_VMS - 1))); do
    instance="${PREFIX}-${i}"
    z=$(vm_zone $i)
    (
        gcloud compute ssh "$instance" --zone="$z" --command="
            pkill -f generate_mc_teacher.py 2>/dev/null || true
            sleep 1
            rm -rf ~/ofc-pineapple/teacher_results/*
            mkdir -p ~/ofc-pineapple/teacher_results
            echo 'Cleaned up'
        " 2>/dev/null
        echo "  ${instance}: cleaned"
    ) &
done
wait
echo "=== All VMs cleaned ==="

# ---- Step 3: Restart with new settings ----
echo ""
echo "=== Step 3: Starting s200 generation: ${NEW_TOTAL_HANDS} hands, ${NEW_SIMS} sims, ${NUM_VMS} VMs x ${WORKERS_PER_VM} workers ==="
for i in $(seq 0 $((NUM_VMS - 1))); do
    instance="${PREFIX}-${i}"
    z=$(vm_zone $i)
    vm_hand_start=$((i * NEW_HANDS_PER_VM))
    (
        gcloud compute ssh "$instance" --zone="$z" --command="
            cd ~/ofc-pineapple
            for w in \$(seq 0 $((WORKERS_PER_VM - 1))); do
                hand_start=\$((${vm_hand_start} + w * ${NEW_HANDS_PER_WORKER}))
                hand_end=\$((hand_start + ${NEW_HANDS_PER_WORKER}))
                shard_id=\$((${i} * ${WORKERS_PER_VM} + w))
                outfile=\"teacher_results/mc_s${NEW_SIMS}_shard\${shard_id}.jsonl\"
                nohup python -u generate_mc_teacher.py \
                    --n-hands ${NEW_TOTAL_HANDS} --seed ${NEW_SEED} --sims ${NEW_SIMS} \
                    --hand-start \${hand_start} --hand-end \${hand_end} \
                    --output \${outfile} \
                    > teacher_results/shard\${shard_id}.log 2>&1 &
                echo \"  Worker \${w}: shard \${shard_id}, hands [\${hand_start}, \${hand_end})\"
            done
            echo \"Started ${WORKERS_PER_VM} workers on VM ${i}\"
        " 2>/dev/null
        echo "  ${instance}: ${WORKERS_PER_VM} workers started"
    ) &
done
wait
echo "=== All shards started (${NUM_VMS} VMs x ${WORKERS_PER_VM} workers = $((NUM_VMS * WORKERS_PER_VM)) shards) ==="
echo ""
echo "============================================"
echo "  ${NUM_VMS} VMs running s${NEW_SIMS}."
echo "  Each worker: ${NEW_HANDS_PER_WORKER} hands"
echo "  Check progress with:"
echo "    ./gcp_mc_teacher.sh status"
echo "  (after updating SIMS=200 in the script)"
echo "============================================"
