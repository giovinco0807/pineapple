#!/bin/bash
# Launch Spot VMs for Self-Play Data Generation
#
# Settings: N_GAMES=10,000 per VM, N_VMS=20 (Total: 200,000 games, ~1.6 million hands)

set -euo pipefail

PROJECT="ofc-solver-485418"
MACHINE="e2-highcpu-32"
N_VMS=20
N_GAMES=10000
N2=3
N3=3
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

echo "=== Launching $N_VMS Spot VMs for Self-Play ==="
echo "  Machine: $MACHINE"
echo "  Games/VM: $N_GAMES"
echo "  Total games: $((N_GAMES * N_VMS))"
echo "  MC Settings: n2=$N2, n3=$N3, n4=$N4"
echo ""

for i in $(seq 0 $((N_VMS - 1))); do
    ZONE="${ZONES[$i]}"
    VM_NAME="selfplay-worker-${i}"

    echo "[$((i+1))/$N_VMS] Creating $VM_NAME in $ZONE..."

    gcloud compute instances create "$VM_NAME" \
        --project="$PROJECT" \
        --zone="$ZONE" \
        --machine-type="$MACHINE" \
        --provisioning-model=SPOT \
        --instance-termination-action=STOP \
        --no-restart-on-failure \
        --metadata="VM_ID=$i,N_GAMES=$N_GAMES,N2=$N2,N3=$N3,N4=$N4" \
        --metadata-from-file=startup-script=ai/gcp_selfplay.sh \
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
echo "Monitor with: gcloud compute instances list --filter='name~selfplay-worker'"
echo "Check logs:   gcloud compute ssh selfplay-worker-0 --zone=us-central1-a -- tail -f /var/log/syslog"
