#!/bin/bash
# Restart s200 generation - using absolute python path
set -e

ZONES=("us-central1-a" "us-central1-b" "us-central1-c" "us-central1-f" "us-east1-b" "us-east1-c" "us-east1-d" "us-east4-a" "us-east4-b" "us-east4-c")
PREFIX="ofc-teacher"
NUM_VMS=30
WORKERS_PER_VM=8

NEW_TOTAL_HANDS=4800
NEW_SIMS=200
NEW_SEED=42
NEW_HANDS_PER_VM=$((NEW_TOTAL_HANDS / NUM_VMS))
NEW_HANDS_PER_WORKER=$((NEW_HANDS_PER_VM / WORKERS_PER_VM))

vm_zone() { echo "${ZONES[$(($1 % ${#ZONES[@]}))]}" ; }

echo "=== Restarting s200 with absolute python path ==="
for i in $(seq 0 $((NUM_VMS - 1))); do
    instance="${PREFIX}-${i}"
    z=$(vm_zone $i)
    vm_hand_start=$((i * NEW_HANDS_PER_VM))
    (
        gcloud compute ssh "$instance" --zone="$z" --command="
            pkill -f generate_mc_teacher.py 2>/dev/null || true
            sleep 1
            rm -rf ~/ofc-pineapple/teacher_results/*
            mkdir -p ~/ofc-pineapple/teacher_results
            cd ~/ofc-pineapple
            export PATH=\\\$HOME/.cargo/bin:\\\$PATH
            PYTHON=\\\$HOME/venv/bin/python
            for w in \$(seq 0 $((WORKERS_PER_VM - 1))); do
                hand_start=\$((${vm_hand_start} + w * ${NEW_HANDS_PER_WORKER}))
                hand_end=\$((hand_start + ${NEW_HANDS_PER_WORKER}))
                shard_id=\$((${i} * ${WORKERS_PER_VM} + w))
                outfile=\"teacher_results/mc_s${NEW_SIMS}_shard\${shard_id}.jsonl\"
                nohup \\\$PYTHON -u generate_mc_teacher.py \
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
echo "=== All shards restarted ==="
