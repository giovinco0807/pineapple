#!/usr/bin/env bash
# Pass-2 supervisor, preemption-hardened.
#
# The first supervisor died two ways: a transient gcloud failure made its
# wait loop read "no instances" as "wave complete", and it launched the next
# wave while the previous one still held the Spot quota.  This one drives
# each run to its done-marker count with skip-done relaunches every cycle,
# never trusts a single empty listing, and starts a wave only after the
# previous wave's instances are actually gone.
set -u
cd "C:/Users/Owner/.gemini/antigravity/scratch/ofc-pineapple"
BUCKET=pokerhu-ofc-solver-485418-training

done_count() { gcloud storage ls "gs://$BUCKET/joker-fleet/runs/$1/done/" 2>/dev/null | wc -l; }

drive_run() {
  # street run_id shards roots batch seed_base watchdog extra...
  local street=$1 run=$2 shards=$3 roots=$4 batch=$5 seed=$6 dog=$7; shift 7
  local extra="$*"
  local cycles=0
  while [ "$(done_count "$run")" -lt "$shards" ]; do
    cycles=$(( cycles + 1 ))
    if [ "$cycles" -gt 40 ]; then
      echo "GIVING UP on $run after 40 cycles"
      return 1
    fi
    # Relaunch whatever is neither done nor currently running.  skip-done
    # covers finished shards; existing instance names collide harmlessly
    # (creation fails, the shard keeps running).
    python -m ai.tutor.fleet.launch_joker_fleet --street "$street" \
      --run-id "$run" --shards "$shards" --roots-per-shard "$roots" \
      --batch "$batch" --seed-base "$seed" --watchdog-seconds "$dog" \
      --skip-done --extra-args "$extra" 2>&1 | grep -E "^launched|^skip-done|FAILED" || true
    sleep 1200
    echo "$run: done=$(done_count "$run")/$shards $(date +%H:%M)"
  done
  echo "$run COMPLETE"
}

wave_clear() {
  # Wait until no jk- instances remain (quota free), tolerating flaky reads.
  local quiet=0
  while [ "$quiet" -lt 2 ]; do
    if OUT=$(gcloud compute instances list --filter="name~jk- AND status=RUNNING" \
        --format="value(name)" 2>/dev/null); then
      if [ -z "$OUT" ]; then quiet=$(( quiet + 1 )); else quiet=0; fi
    fi
    sleep 120
  done
}

for C in 15 16 17; do
  drive_run t2 "t2w1-c$C" 10 600 50 "21${C}000000" 12600 \
    "--t3-samples 50 --t4-draw-sample 100 --force-opp-count $C"
done
echo "=== WAVE1 T2 DONE $(date)"
wave_clear
for C in 15 16 17; do
  drive_run t1 "t1w1-c$C" 10 400 25 "22${C}000000" 14400 \
    "--t2-samples 20 --t3-samples 10 --t4-draw-sample 60 --force-opp-count $C"
done
echo "=== WAVE2 T1 DONE $(date)"
wave_clear
for C in 15 16 17; do
  drive_run t0 "t0w1-c$C" 10 60 10 "23${C}000000" 14400 \
    "--t1-model /var/lib/joker-labelgen/models/t1_evaluator.bin --t1-samples 6 --t2-samples 4 --t3-samples 3 --t4-draw-sample 40 --force-opp-count $C"
done
echo "=== PASS2 FLEET ALL DONE $(date)"
