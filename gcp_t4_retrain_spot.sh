#!/bin/bash
# ============================================================================
# GCP Spot GPU T4 Oracle Retraining
# ============================================================================
# Usage:
#   ./gcp_t4_retrain_spot.sh create
#   ./gcp_t4_retrain_spot.sh upload
#   ./gcp_t4_retrain_spot.sh setup
#   ./gcp_t4_retrain_spot.sh run
#   ./gcp_t4_retrain_spot.sh status
#   ./gcp_t4_retrain_spot.sh download
#   ./gcp_t4_retrain_spot.sh delete
#   ./gcp_t4_retrain_spot.sh all
# ============================================================================

set -e

export CLOUDSDK_CORE_ACCOUNT="giovinco.080807@gmail.com"

PROJECT="ofc-solver-485418"
ZONE="us-east1-c"
INSTANCE="ofc-t4train-spot-0"
MACHINE_TYPE="n1-standard-4"
GPU_TYPE="nvidia-tesla-t4"
GPU_COUNT=1
IMAGE_FAMILY="pytorch-2-9-cu129-ubuntu-2204-nvidia-580"
IMAGE_PROJECT="deeplearning-platform-release"
DISK_SIZE="150GB"

GCS_BUCKET="gs://ofc-solver-results"
GCS_DATASET="t4_dataset_v2"
GCS_RESULTS="t4_oracle_v2_retrain"

LOCAL_ROOT="$(cd "$(dirname "$0")" && pwd)"
LOCAL_DATA_DIR="${LOCAL_ROOT}/ai/data/t4_dataset_v2"
LOCAL_RESULT_DIR="${LOCAL_ROOT}/ai/data/t4_oracle_v2_retrain_gcp"

REMOTE_ROOT="~/ofc-pineapple"
REMOTE_DATA_DIR="${REMOTE_ROOT}/ai/data/t4_dataset_v2"
REMOTE_SAVE_DIR="${REMOTE_ROOT}/ai/data/t4_oracle_v2_retrain"

create_vm() {
    echo "=== Creating Spot GPU VM: ${INSTANCE} ==="
    gcloud compute instances create "${INSTANCE}" \
        --project="${PROJECT}" \
        --zone="${ZONE}" \
        --machine-type="${MACHINE_TYPE}" \
        --image-family="${IMAGE_FAMILY}" \
        --image-project="${IMAGE_PROJECT}" \
        --boot-disk-size="${DISK_SIZE}" \
        --boot-disk-type="pd-balanced" \
        --accelerator="type=${GPU_TYPE},count=${GPU_COUNT}" \
        --maintenance-policy=TERMINATE \
        --provisioning-model=SPOT \
        --instance-termination-action=STOP \
        --scopes="default,storage-rw"
    echo "=== Waiting 60s for boot + GPU driver init ==="
    sleep 60
}

upload_code_and_data() {
    echo "=== Packing training code ==="
    cd "${LOCAL_ROOT}"
    TAR_PATH="/tmp/ofc_t4_train_code.tar.gz"
    tar czf "${TAR_PATH}" \
        ai/__init__.py \
        ai/engine/__init__.py \
        ai/engine/encoding.py \
        ai/engine/action_space.py \
        ai/training/__init__.py \
        ai/training/train_t4_oracle.py \
        ai/training/evaluate_t4_oracle.py

    echo "=== Uploading code to VM ==="
    gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --command="mkdir -p ${REMOTE_ROOT}" 2>/dev/null
    gcloud compute scp "${TAR_PATH}" "${INSTANCE}:/tmp/ofc_t4_train_code.tar.gz" --zone="${ZONE}" 2>/dev/null
    gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --command="
        cd ${REMOTE_ROOT}
        tar xzf /tmp/ofc_t4_train_code.tar.gz
        rm /tmp/ofc_t4_train_code.tar.gz
    " 2>/dev/null

    echo "=== Uploading T4 dataset to GCS ==="
    gsutil -m rsync -r "${LOCAL_DATA_DIR}" "${GCS_BUCKET}/${GCS_DATASET}"

    echo "=== Downloading T4 dataset from GCS to VM ==="
    gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --command="
        mkdir -p ${REMOTE_DATA_DIR}
        gsutil -m rsync -r ${GCS_BUCKET}/${GCS_DATASET} ${REMOTE_DATA_DIR}
    "
}

setup_vm() {
    echo "=== Setting up VM ==="
    gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --command="
        set -e
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
        python3 -m pip install -q numpy torch
        cd ${REMOTE_ROOT}
        touch ai/__init__.py ai/training/__init__.py
        python3 - << 'PY'
import torch, numpy
print('torch', torch.__version__)
print('cuda', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu', torch.cuda.get_device_name(0))
print('numpy', numpy.__version__)
PY
    "
}

run_training() {
    echo "=== Starting T4 retraining on ${INSTANCE} ==="
    gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --command="
        cd ${REMOTE_ROOT}
        mkdir -p ${REMOTE_SAVE_DIR}
        export PYTHONUNBUFFERED=1
        nohup python3 -u ai/training/train_t4_oracle.py \
            --data-dir ai/data/t4_dataset_v2 \
            --save-dir ai/data/t4_oracle_v2_retrain \
            --epochs 240 \
            --batch-size 4096 \
            --lr 3e-4 \
            --temp-start 3.0 \
            --temp-end 1.0 \
            --value-weight 0.2 \
            --eval-every 5 \
            --device cuda \
            > ai/data/t4_oracle_v2_retrain/train.log 2>&1 &
        echo \"Training PID: \$!\"
    "
}

status_vm() {
    echo "=== VM status ==="
    gcloud compute instances describe "${INSTANCE}" --zone="${ZONE}" \
        --format="table(name,status,machineType.basename(),scheduling.provisioningModel)"
    echo
    echo "=== Training log tail ==="
    gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --command="
        tail -80 ${REMOTE_SAVE_DIR}/train.log 2>/dev/null || echo 'train.log not found yet'
        echo
        pgrep -af train_t4_oracle.py || true
    "
}

download_results() {
    echo "=== Uploading results from VM to GCS ==="
    gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --command="
        gsutil -m rsync -r ${REMOTE_SAVE_DIR} ${GCS_BUCKET}/${GCS_RESULTS}
    "

    echo "=== Downloading results to local ==="
    mkdir -p "${LOCAL_RESULT_DIR}"
    gsutil -m rsync -r "${GCS_BUCKET}/${GCS_RESULTS}" "${LOCAL_RESULT_DIR}"
    echo "Saved to ${LOCAL_RESULT_DIR}"
}

delete_vm() {
    echo "=== Deleting VM: ${INSTANCE} ==="
    gcloud compute instances delete "${INSTANCE}" --zone="${ZONE}" --quiet
}

case "${1:-status}" in
    create) create_vm ;;
    upload) upload_code_and_data ;;
    setup) setup_vm ;;
    run) run_training ;;
    status) status_vm ;;
    download) download_results ;;
    delete) delete_vm ;;
    all)
        create_vm
        upload_code_and_data
        setup_vm
        run_training
        ;;
    *)
        echo "Usage: $0 {create|upload|setup|run|status|download|delete|all}"
        exit 1
        ;;
esac
