#!/bin/bash
# ============================================================================
# GCP Benchmark Setup Script
# ============================================================================
# Creates a GCP VM, installs dependencies, builds Rust FL solver, and uploads
# project code + model files for running eval_mcts.py benchmarks.
#
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - GCP project: ofc-solver-485418
#   - Model files present locally
#
# Usage:
#   ./gcp_setup_bench.sh create    # Create VM
#   ./gcp_setup_bench.sh upload    # Upload code + models
#   ./gcp_setup_bench.sh setup     # Install deps + build Rust on VM
#   ./gcp_setup_bench.sh all       # Do all of the above
#   ./gcp_setup_bench.sh ssh       # SSH into VM
#   ./gcp_setup_bench.sh delete    # Delete VM
# ============================================================================

set -e

PROJECT="ofc-solver-485418"
ZONE="us-central1-a"
INSTANCE="ofc-bench"
MACHINE_TYPE="n2-standard-8"
IMAGE_FAMILY="ubuntu-2204-lts"
IMAGE_PROJECT="ubuntu-os-cloud"
DISK_SIZE="50GB"

# Local project root (adjust if running from elsewhere)
LOCAL_ROOT="$(cd "$(dirname "$0")" && pwd)"
REMOTE_DIR="/home/$(gcloud config get-value account 2>/dev/null | sed 's/@.*//')/ofc-pineapple"
# Fallback: use $USER
REMOTE_USER="${USER:-owner}"
REMOTE_DIR="/home/${REMOTE_USER}/ofc-pineapple"

# ── Create VM ──────────────────────────────────────────────────────────────

create_vm() {
    echo "=== Creating VM: ${INSTANCE} ==="
    gcloud compute instances create "$INSTANCE" \
        --project="$PROJECT" \
        --zone="$ZONE" \
        --machine-type="$MACHINE_TYPE" \
        --image-family="$IMAGE_FAMILY" \
        --image-project="$IMAGE_PROJECT" \
        --boot-disk-size="$DISK_SIZE" \
        --boot-disk-type="pd-ssd" \
        --scopes="default"
    echo "=== VM created ==="
    echo "Waiting 30s for SSH to become available..."
    sleep 30
}

# ── Upload code + models ──────────────────────────────────────────────────

upload() {
    echo "=== Uploading code and models ==="

    # Create remote directory structure
    gcloud compute ssh "$INSTANCE" --zone="$ZONE" --command="
        mkdir -p ~/ofc-pineapple/ai/engine
        mkdir -p ~/ofc-pineapple/ai/mcts
        mkdir -p ~/ofc-pineapple/ai/models
        mkdir -p ~/ofc-pineapple/ai/config
        mkdir -p ~/ofc-pineapple/ai/rust_solver/fl_solver/src
        mkdir -p ~/ofc-pineapple/ai/rust_solver/ofc_core/src
        mkdir -p ~/ofc-pineapple/ai/rust_solver/backward/src
        mkdir -p ~/ofc-pineapple/ai/models/expectimax_bc_v4
        mkdir -p ~/ofc-pineapple/ai/models/value_v5
        mkdir -p ~/ofc-pineapple/ai/models/value_v6
    "

    # ── Pack code into tar and upload (avoids Windows scp glob issues) ──

    echo "  Packing code archive..."
    cd "$LOCAL_ROOT"

    # Create a tarball of all needed files
    TAR_PATH="/tmp/ofc_bench_code.tar.gz"
    tar czf "$TAR_PATH" \
        eval_mcts.py \
        eval_baseline.py \
        eval_hybrid.py \
        ai/__init__.py \
        ai/rust_solver_wrapper.py \
        ai/engine/__init__.py \
        ai/engine/encoding.py \
        ai/engine/action_space.py \
        ai/engine/game_engine.py \
        ai/engine/scoring.py \
        ai/mcts/__init__.py \
        ai/mcts/mcts.py \
        ai/mcts/rollout_evaluator.py \
        ai/mcts/multi_turn_mcts.py \
        ai/mcts/ofc_mcts.py \
        ai/models/__init__.py \
        ai/models/networks.py \
        ai/config/fl_ev.json \
        ai/rust_solver/Cargo.toml \
        ai/rust_solver/Cargo.lock \
        ai/rust_solver/fl_solver/Cargo.toml \
        ai/rust_solver/fl_solver/src/main.rs \
        ai/rust_solver/ofc_core/Cargo.toml \
        ai/rust_solver/ofc_core/src/lib.rs \
        ai/rust_solver/backward/Cargo.toml \
        ai/rust_solver/backward/src/main.rs \
        gcp_run_bench.sh

    echo "  Uploading code archive..."
    gcloud compute scp "$TAR_PATH" \
        "$INSTANCE:/tmp/ofc_bench_code.tar.gz" --zone="$ZONE"

    gcloud compute ssh "$INSTANCE" --zone="$ZONE" --command="
        cd ~/ofc-pineapple && tar xzf /tmp/ofc_bench_code.tar.gz
        rm /tmp/ofc_bench_code.tar.gz
    "

    # ── Upload model files ──

    echo "  Uploading model files..."

    # BC policy v4
    if [ -f "$LOCAL_ROOT/ai/models/expectimax_bc_v4/bc_policy_best.pt" ]; then
        gcloud compute scp \
            "$LOCAL_ROOT/ai/models/expectimax_bc_v4/bc_policy_best.pt" \
            "$INSTANCE:/home/${REMOTE_USER}/ofc-pineapple/ai/models/expectimax_bc_v4/bc_policy_best.pt" \
            --zone="$ZONE"
        echo "    Uploaded expectimax_bc_v4/bc_policy_best.pt"
    else
        echo "    WARNING: expectimax_bc_v4/bc_policy_best.pt not found!"
    fi

    # Value v5
    if [ -f "$LOCAL_ROOT/ai/models/value_v5/value_best.pt" ]; then
        gcloud compute scp \
            "$LOCAL_ROOT/ai/models/value_v5/value_best.pt" \
            "$INSTANCE:/home/${REMOTE_USER}/ofc-pineapple/ai/models/value_v5/value_best.pt" \
            --zone="$ZONE"
        echo "    Uploaded value_v5/value_best.pt"
    else
        echo "    WARNING: value_v5/value_best.pt not found!"
    fi

    # Value v6
    if [ -f "$LOCAL_ROOT/ai/models/value_v6/value_best.pt" ]; then
        gcloud compute scp \
            "$LOCAL_ROOT/ai/models/value_v6/value_best.pt" \
            "$INSTANCE:/home/${REMOTE_USER}/ofc-pineapple/ai/models/value_v6/value_best.pt" \
            --zone="$ZONE"
        echo "    Uploaded value_v6/value_best.pt"
    else
        echo "    WARNING: value_v6/value_best.pt not found!"
    fi

    echo "=== Upload complete ==="
}

# ── Install deps + build Rust on VM ──────────────────────────────────────

setup_vm() {
    echo "=== Setting up VM ==="
    gcloud compute ssh "$INSTANCE" --zone="$ZONE" --command="
        set -e

        echo '--- Installing system packages ---'
        sudo apt-get update -qq
        sudo apt-get install -y -qq python3.11 python3.11-venv python3.11-dev curl build-essential

        echo '--- Creating Python venv ---'
        python3.11 -m venv ~/venv
        source ~/venv/bin/activate
        pip install --upgrade pip

        echo '--- Installing Python packages ---'
        pip install torch --index-url https://download.pytorch.org/whl/cpu
        pip install numpy

        echo '--- Verifying Python setup ---'
        python -c 'import torch; print(f\"PyTorch {torch.__version__} (CPU)\")'
        python -c 'import numpy; print(f\"NumPy {numpy.__version__}\")'

        echo '--- Installing Rust ---'
        if ! command -v cargo &>/dev/null; then
            curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
            source \"\$HOME/.cargo/env\"
        fi
        rustc --version
        cargo --version

        echo '--- Building Rust FL solver ---'
        cd ~/ofc-pineapple/ai/rust_solver
        cargo build --release -p fl_solver
        ls -la target/release/fl_solver
        echo '--- Rust FL solver built ---'

        echo '--- Patching rust_solver_wrapper.py for Linux ---'
        cd ~/ofc-pineapple
        sed -i 's/fl_solver\\.exe/fl_solver/' ai/rust_solver_wrapper.py

        echo '--- Making run script executable ---'
        chmod +x gcp_run_bench.sh

        echo '=== Setup complete ==='
    "
}

# ── SSH into VM ──────────────────────────────────────────────────────────

ssh_vm() {
    gcloud compute ssh "$INSTANCE" --zone="$ZONE"
}

# ── Delete VM ────────────────────────────────────────────────────────────

delete_vm() {
    echo "=== Deleting VM: ${INSTANCE} ==="
    gcloud compute instances delete "$INSTANCE" \
        --project="$PROJECT" \
        --zone="$ZONE" \
        --quiet
    echo "=== VM deleted ==="
}

# ── Download results ─────────────────────────────────────────────────────

download() {
    echo "=== Downloading results ==="
    mkdir -p "$LOCAL_ROOT/data/bench_results"
    gcloud compute scp "$INSTANCE:/home/${REMOTE_USER}/ofc-pineapple/bench_results/*" \
        "$LOCAL_ROOT/data/bench_results/" --zone="$ZONE" --recurse
    echo "=== Results downloaded to data/bench_results/ ==="
}

# ── Main ─────────────────────────────────────────────────────────────────

case "${1:-help}" in
    create)
        create_vm
        ;;
    upload)
        upload
        ;;
    setup)
        setup_vm
        ;;
    all)
        create_vm
        upload
        setup_vm
        echo ""
        echo "============================================"
        echo "  VM ready! SSH in and run benchmarks:"
        echo "    ./gcp_setup_bench.sh ssh"
        echo "    cd ~/ofc-pineapple"
        echo "    ./gcp_run_bench.sh"
        echo "============================================"
        ;;
    ssh)
        ssh_vm
        ;;
    delete)
        delete_vm
        ;;
    download)
        download
        ;;
    *)
        echo "Usage: $0 {create|upload|setup|all|ssh|delete|download}"
        echo ""
        echo "  create   - Create GCP VM (n2-standard-8, Ubuntu 22.04)"
        echo "  upload   - Upload code + model files to VM"
        echo "  setup    - Install Python, PyTorch(CPU), Rust, build FL solver"
        echo "  all      - Create + upload + setup (full workflow)"
        echo "  ssh      - SSH into VM"
        echo "  delete   - Delete VM"
        echo "  download - Download benchmark results from VM"
        ;;
esac
