$env:CLOUDSDK_CORE_ACCOUNT="giovinco.080807@gmail.com"
$PROJECT="ofc-solver-485418"
$ZONE="us-east1-d"
$INSTANCE="ofc-t3gen-0"

Write-Host "Packing Rust T3 generator..."
tar czf ofc_t3_code.tar.gz ai/rust_solver ai/training/convert_t3_rust_jsonl.py

Write-Host "Uploading code..."
gcloud compute ssh $INSTANCE --zone=$ZONE --command="mkdir -p /home/Owner/ofc-pineapple/ai/data/t4_oracle_v2"
gcloud compute scp ofc_t3_code.tar.gz ${INSTANCE}:/home/Owner/ofc_t3_code.tar.gz --zone=$ZONE
gcloud compute ssh $INSTANCE --zone=$ZONE --command="cd /home/Owner/ofc-pineapple && tar xzf /home/Owner/ofc_t3_code.tar.gz && rm /home/Owner/ofc_t3_code.tar.gz"

Write-Host "Uploading latest T4 ONNX..."
gsutil cp ai/data/t4_oracle_v2/t4_oracle.onnx gs://ofc-solver-results/t4_oracle_v2/t4_oracle.onnx
gcloud compute ssh $INSTANCE --zone=$ZONE --command="gsutil cp gs://ofc-solver-results/t4_oracle_v2/t4_oracle.onnx /home/Owner/ofc-pineapple/ai/data/t4_oracle_v2/t4_oracle.onnx"

Write-Host "Setting up Rust + NumPy..."
$setup = @"
set -e
if ! command -v cargo >/dev/null 2>&1; then
  curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal
fi
source "`$HOME/.cargo/env"
pip install -q numpy 2>/dev/null || true
cd /home/Owner/ofc-pineapple
cargo build -p t3_generator --release --manifest-path ai/rust_solver/Cargo.toml
"@
$setup | Out-File -Encoding ASCII gcp_t3_setup.sh
gcloud compute scp gcp_t3_setup.sh ${INSTANCE}:/home/Owner/gcp_t3_setup.sh --zone=$ZONE
gcloud compute ssh $INSTANCE --zone=$ZONE --command="bash /home/Owner/gcp_t3_setup.sh"

Write-Host "Running Rust T3 Data Generation..."
gcloud compute ssh $INSTANCE --zone=$ZONE --command="cd /home/Owner/ofc-pineapple && mkdir -p ai/data/t3_results/bb && nohup bash -lc './ai/rust_solver/target/release/t3_generator --states 100000 --mode bb --btn-samples 500 --onnx-model ai/data/t4_oracle_v2/t4_oracle.onnx --output ai/data/t3_results/bb/t3_teacher.jsonl --log-interval 100 && python3 ai/training/convert_t3_rust_jsonl.py --input ai/data/t3_results/bb/t3_teacher.jsonl --output-dir ai/data/t3_results/bb --chunk-size 10000' > ai/data/t3_results/bb_gen.log 2>&1 &"
Write-Host "Done."
