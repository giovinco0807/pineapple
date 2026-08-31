#!/bin/bash
# GCP deployment script for T2 bottom-up data generation
# Usage:
#   ./gcp_t2_gen.sh setup    - Install dependencies
#   ./gcp_t2_gen.sh run START END  - Run generation for file range [START, END)
#
# Example for 4 VMs with 686 files:
#   VM1: ./gcp_t2_gen.sh run 0 172
#   VM2: ./gcp_t2_gen.sh run 172 344
#   VM3: ./gcp_t2_gen.sh run 344 516
#   VM4: ./gcp_t2_gen.sh run 516 686

set -e

WORKDIR=~/t2gen
N_SAMPLES=30
N_WORKERS=30  # n2-highcpu-32 has 32 vCPUs
SEED=42

setup() {
    echo "=== Setting up T2 data generation ==="
    mkdir -p $WORKDIR
    cd $WORKDIR

    # Install Python and pip
    sudo apt-get update -qq
    sudo apt-get install -y -qq python3 python3-pip python3-venv > /dev/null 2>&1

    # Create venv and install deps
    python3 -m venv venv
    source venv/bin/activate
    pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
    pip install --quiet numpy

    # Extract code
    tar xzf ~/t2_gen_code.tar.gz

    # Extract data
    echo "Extracting JSON data..."
    tar xzf ~/expectimax_v3_data.tar.gz

    # Copy BC model
    cp ~/bc_policy_best.pt .

    echo "=== Setup complete ==="
    echo "Files:"
    ls -la data/expectimax_results_v3/*.json | wc -l
    echo "JSON files ready"
}

run() {
    START=$1
    END=$2
    cd $WORKDIR
    source venv/bin/activate

    echo "=== Running T2 generation: files [$START, $END) ==="
    echo "Workers: $N_WORKERS, Samples: $N_SAMPLES"

    PYTHONUNBUFFERED=1 python3 -u -m ai.training.generate_bottomup_data \
        --turn 2 \
        --n-samples $N_SAMPLES \
        --bc-t3 bc_policy_best.pt \
        --json-dir data/expectimax_results_v3 \
        --save output_t2 \
        --workers $N_WORKERS \
        --file-start $START \
        --file-end $END \
        --seed $SEED

    echo "=== Done ==="
    echo "Output:"
    ls -la output_t2/
}

case "$1" in
    setup)
        setup
        ;;
    run)
        run $2 $3
        ;;
    *)
        echo "Usage: $0 {setup|run START END}"
        exit 1
        ;;
esac
