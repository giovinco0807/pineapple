#!/bin/bash
# Download all shards from 10 GCP VMs to D:\ofc_data\mc_teacher
# Run from Git Bash on Windows

PROJECT="ofc-solver-485418"
SIMS=50
NUM_VMS=10
WORKERS_PER_VM=8
LOCAL_DEST="/d/ofc_data/mc_teacher"
mkdir -p "$LOCAL_DEST"

echo "=== Downloading shards from all 10 VMs to $LOCAL_DEST ==="
echo ""

vm_zone() {
    local i=$1
    if [ "$i" -eq 0 ] || [ "$i" -ge 8 ]; then
        echo "us-east1-b"
    else
        echo "us-central1-a"
    fi
}

for i in $(seq 0 $((NUM_VMS - 1))); do
    instance="ofc-teacher-${i}"
    z=$(vm_zone $i)
    echo "--- VM${i}: ${instance} (${z}) ---"
    
    for w in $(seq 0 $((WORKERS_PER_VM - 1))); do
        sid=$((i * WORKERS_PER_VM + w))
        remote_data="~/ofc-pineapple/teacher_results/mc_s${SIMS}_shard${sid}.jsonl"
        remote_log="~/ofc-pineapple/teacher_results/shard${sid}.log"
        local_data="$LOCAL_DEST/mc_s${SIMS}_shard${sid}.jsonl"
        local_log="$LOCAL_DEST/shard${sid}.log"

        echo -n "  Shard${sid}: "
        
        # Download data file (via SSH cat)
        gcloud compute ssh "$instance" --zone="$z" --project="$PROJECT" \
            --command="cat $remote_data 2>/dev/null" 2>/dev/null > "$local_data"
        
        lines=$(wc -l < "$local_data" 2>/dev/null || echo 0)
        hands=$((lines / 6))
        echo "${lines} records (~${hands} hands)"
        
        # Download log file
        gcloud compute ssh "$instance" --zone="$z" --project="$PROJECT" \
            --command="cat $remote_log 2>/dev/null" 2>/dev/null > "$local_log"
    done
    echo ""
done

echo "=== Download complete. Checking totals ==="
total_lines=0
for f in "$LOCAL_DEST"/mc_s${SIMS}_shard*.jsonl; do
    if [ -f "$f" ] && [ -s "$f" ]; then
        lines=$(wc -l < "$f" 2>/dev/null || echo 0)
        total_lines=$((total_lines + lines))
    fi
done
total_hands=$((total_lines / 6))
echo "Total records: $total_lines"
echo "Total hands:   $total_hands / 10000 ($(( total_hands * 100 / 10000 ))%)"

echo ""
echo "=== Merging all shards ==="
cat "$LOCAL_DEST"/mc_s${SIMS}_shard*.jsonl > "$LOCAL_DEST/mc_s${SIMS}_merged.jsonl" 2>/dev/null
merged=$(wc -l < "$LOCAL_DEST/mc_s${SIMS}_merged.jsonl")
echo "Merged file: $merged records -> $LOCAL_DEST/mc_s${SIMS}_merged.jsonl"
echo ""
echo "Done! Files saved to: $LOCAL_DEST"
