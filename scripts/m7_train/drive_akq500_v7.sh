#!/usr/bin/env bash
# Drive the akq500 v7 run: report every ten minutes, refill what Spot took,
# and -- the part this run exists to add -- prove early that the workers are
# COMPUTING, not merely alive.
#
# The v6 attempt put 53 VMs in the air against a runtime whose worker could not
# read an explicit-roots plan. Every one died in its startup script with
# "plan is missing fields: ['behavior_seed_offset', 'hand_seed_base']", and a
# watch that counted receipts and instances could not tell that from healthy
# work: 53 alive and 0 published is exactly what a healthy fleet looks like for
# the first ~2.9 hours. An hour of compute was billed for nothing. So the first
# check here reads a worker's serial console and looks for the words a working
# worker prints; silence about failure is not evidence of progress.
set -uo pipefail

RUN=t0first-akq500-1024p-v7
TOTAL=63
ZONE=us-west1-a
PKG=/home/wner/ofc-labelgen-m7v7/package
WP=$PKG/worker_plan_t0first_akq500_1024p.json
RP=/home/wner/ofc-m7/akq500_runs/run_plan_t0first-akq500-1024p-v7.json
SA=ofc-labelgen-worker@ofc-solver-485418.iam.gserviceaccount.com
REPO=/mnt/c/Users/Owner/.gemini/antigravity/scratch/ofc-pineapple/regular-ofc-pineapple

G="C:/Users/Owner/AppData/Local/Google/Cloud SDK/google-cloud-sdk/bin/gcloud.cmd"
export CLOUDSDK_CONFIG="C:/Users/Owner/AppData/Roaming/gcloud"
BUCKET="gs://pokerhu-ofc-solver-485418-training/labelgen/$RUN"
STALL_MIN=240

done_shards() {
  "$G" storage ls "$BUCKET/shards/*/files/SHARD_DONE.json" 2>/dev/null \
    | sed -n 's#.*/shards/\([0-9]*\)/files/SHARD_DONE.json.*#\1#p' | tr -d '\r'
}
# Space-free arguments only: this gcloud is a .cmd reached through git-bash, and
# an argument containing a space breaks the quoting around the interpreter path,
# fails to stderr, and returns an EMPTY stdout that reads as "nothing running".
instances() {
  "$G" compute instances list --format="value(name,status)" 2>/dev/null \
    | tr -d '\r' | awk -v r="$RUN" 'index($1,r)>0 {print $1, $2}'
}
relaunch() {
  local token
  token=$("$G" auth print-access-token 2>/dev/null | head -1 | tr -d '\r\n ')
  if [ ${#token} -lt 50 ]; then echo "TOKEN FAILED -- needs a human"; return 1; fi
  wsl.exe -e bash -c "cd $REPO && GOOGLE_OAUTH_ACCESS_TOKEN='$token' \
    CLOUDSDK_CONFIG=/mnt/c/Users/Owner/AppData/Roaming/gcloud PYTHONPATH=src \
    /home/wner/ofc-m31-fqv1-venv/bin/python3 -m ofc_regular.hu_m31_label_gen_gcp_execute_v1 \
    --phase execute --plan-receipt $RP --worker-plan $WP --package-dir $PKG \
    --service-account $SA --only-missing --allow-writes" 2>&1 | tail -1
}

# --- the check the last run lacked -------------------------------------------
sleep 240
probe=$(instances | awk '$2=="RUNNING" {print $1; exit}')
if [ -z "$probe" ]; then
  echo "$(date +%H:%M) akq500 v7: nothing RUNNING four minutes in -- needs a human"
  exit 1
fi
log=$("$G" compute instances get-serial-port-output "$probe" --zone "$ZONE" 2>/dev/null | tr -d '\r')
if echo "$log" | grep -qiE "missing fields|worker exited [1-9]|Traceback|startup-script.*failed"; then
  echo "$(date +%H:%M) akq500 v7 WORKERS DYING: $(echo "$log" | grep -iE 'missing fields|worker exited|Traceback' | tail -1)"
  echo "  delete the fleet before it bills for nothing"
  exit 1
fi
if echo "$log" | grep -qiE "positions to generate|produced .* new positions"; then
  echo "$(date +%H:%M) akq500 v7: workers COMPUTING -- $(echo "$log" | grep -iE 'positions to generate' | tail -1)"
else
  echo "$(date +%H:%M) akq500 v7: $probe alive but not yet printing progress; watching"
fi

last_receipt_at=$(date +%s)
last_n=-1
while true; do
  mapfile -t published < <(done_shards)
  n=${#published[@]}
  mapfile -t rows < <(instances)
  running=0
  reclaim=""
  for row in "${rows[@]}"; do
    name=${row%% *}; status=${row##* }
    shard=${name##*-}
    if [ "$status" = "RUNNING" ]; then
      running=$((running + 1))
    else
      for d in "${published[@]}"; do
        [ "$shard" = "$d" ] && reclaim="$reclaim $name" && break
      done
    fi
  done

  now=$(date +%s)
  [ "$n" -ne "$last_n" ] && { last_receipt_at=$now; last_n=$n; }
  quiet=$(( (now - last_receipt_at) / 60 ))

  # shellcheck disable=SC2086
  [ -n "$reclaim" ] && "$G" compute instances delete $reclaim --zone "$ZONE" --quiet >/dev/null 2>&1

  if [ "$n" -ge "$TOTAL" ]; then
    echo "$(date +%H:%M) akq500 v7 DONE $n/$TOTAL -- deleting every instance"
    for row in "${rows[@]}"; do
      "$G" compute instances delete "${row%% *}" --zone "$ZONE" --quiet >/dev/null 2>&1
    done
    echo "$(date +%H:%M) cleanup finished"
    break
  fi

  note=""
  # Top up whenever shards are unplaced, not only when the fleet is empty.
  # Waiting for zero was wrong here: another project's jobs hold 60 of the
  # region's slots, so this run got ONE instance in and would then have sat
  # behind its own single worker for ~2.9 hours before trying again. The call
  # is idempotent and quota rejections cost nothing, so attempting every cycle
  # simply claims slots the moment they free.
  placed=$((n + running))
  if [ "$placed" -lt "$TOTAL" ]; then
    out=$(relaunch) || { echo "$out"; break; }
    sleep 45
    after=$(instances | awk '$2=="RUNNING"' | grep -c . || true)
    [ "$after" -ne "$running" ] && note=" :: topped up $running -> $after"
    running=$after
  fi
  if [ -z "$note" ] && [ "$quiet" -ge "$STALL_MIN" ]; then
    note=" :: STALL, ${quiet}min with no new receipt (a shard is ~175min)"
    last_receipt_at=$now
  fi
  echo "$(date +%H:%M) akq500 v7: receipts $n/$TOTAL  running $running$note"
  sleep 600
done
