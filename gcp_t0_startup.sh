#!/bin/bash
set -e
cd /tmp

# Read parameters from instance metadata
SEED=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/batch-seed)
HANDS=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/batch-hands)
VMNAME=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/name)
OUTPUT="t0_batch_${VMNAME}.jsonl"

echo "=== T0 Batch: VM=${VMNAME}, seed=${SEED}, hands=${HANDS} ===" | tee /tmp/batch.log

echo "Downloading binary..." | tee -a /tmp/batch.log
gcloud storage cp gs://ofc-solver-485418/cfr_solver ./cfr_solver
chmod +x ./cfr_solver

echo "Starting: ${HANDS} hands, seed ${SEED}, 50 samples, nesting 10,6,3" | tee -a /tmp/batch.log
./cfr_solver t0-batch --hands ${HANDS} --samples 50 --output /tmp/${OUTPUT} --seed ${SEED} --nesting "10,6,3" 2>&1 | tee -a /tmp/batch.log

echo "Uploading results..." | tee -a /tmp/batch.log
gcloud storage cp /tmp/${OUTPUT} gs://ofc-solver-485418/results/hifi/${OUTPUT}
gcloud storage cp /tmp/batch.log gs://ofc-solver-485418/results/hifi/log_${VMNAME}.txt

echo "DONE - shutting down" | tee -a /tmp/batch.log
shutdown -h now
