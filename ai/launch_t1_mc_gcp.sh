#!/bin/bash
# Launch 20 spot VMs for T1 MC data generation
#
# Total target: 10,000 hands (500 hands/VM × 20 VMs)
# Settings: T0=top5, n2=5, n3=2, n4=3
# Estimated time: ~12h/VM
# Estimated cost: ~$72 total (20 × 12h × $0.30/h)

set -euo pipefail

PROJECT="ofc-solver-485418"
MACHINE="e2-highcpu-32"
N_VMS=20
HANDS_PER_VM=500
N2=5
N3=2
N4=3

# Regions to spread across (avoid quota issues)
ZONES=(
    "us-central1-a"
    "us-central1-b"
    "us-central1-c"
    "us-east1-b"
    "us-east1-c"
    "us-east1-d"
    "us-east4-a"
    "us-east4-b"
    "us-east4-c"
    "us-west1-a"
    "us-west1-b"
    "us-west4-a"
    "us-west4-b"
    "europe-west1-b"
    "europe-west1-c"
    "europe-west2-b"
    "europe-west3-a"
    "europe-west4-a"
    "northamerica-northeast1-a"
    "southamerica-east1-a"
)

echo "=== Launching $N_VMS Spot VMs ==="
echo "  Machine: $MACHINE"
echo "  Hands/VM: $HANDS_PER_VM"
echo "  Total hands: $((HANDS_PER_VM * N_VMS))"
echo "  Settings: n2=$N2, n3=$N3, n4=$N4"
echo ""

# First, push code to git
# echo "=== Pushing latest code ==="
# cd "$(dirname "$0")/.."
# git add -A
# git commit -m "GCP T1 MC: top5, n2=$N2, FL_EV corrected" || echo "Nothing to commit"
# git push || echo "Push failed, continuing..."

for i in $(seq 0 $((N_VMS - 1))); do
    ZONE="${ZONES[$i]}"
    VM_NAME="t1-mc-${i}"

    echo "[$((i+1))/$N_VMS] Creating $VM_NAME in $ZONE..."

    gcloud compute instances create "$VM_NAME" \
        --project="$PROJECT" \
        --zone="$ZONE" \
        --machine-type="$MACHINE" \
        --provisioning-model=SPOT \
        --instance-termination-action=STOP \
        --no-restart-on-failure \
        --metadata="VM_ID=$i,N_HANDS=$HANDS_PER_VM,N2=$N2,N3=$N3,N4=$N4" \
        --metadata-from-file=startup-script=ai/gcp_t1_mc.sh \
        --scopes=storage-rw \
        --boot-disk-size=30GB \
        --image-family=debian-12 \
        --image-project=debian-cloud \
        --no-address \
        2>&1 | tail -1 &

    # Small delay to avoid API rate limits
    sleep 2
done

wait
echo ""
echo "=== All $N_VMS VMs launched ==="
echo "Monitor with: gcloud compute instances list --filter='name~t1-mc'"
echo "Check logs:   gcloud compute ssh t1-mc-0 --zone=us-central1-a -- tail -f /var/log/syslog"
