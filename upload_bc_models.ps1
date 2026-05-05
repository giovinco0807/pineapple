$Bucket = "gs://ofc-solver-485418/ofc_rl_output"
Write-Host "Creating bucket $Bucket (will ignore if already exists)..."
gcloud storage buckets create $Bucket --project=ofc-solver-485418 --location=asia-northeast1

Write-Host "Uploading BC models to $Bucket/models/checkpoints/ ..."
gsutil -m rsync -r ai/models/checkpoints $Bucket/models/checkpoints/
Write-Host "Done!"
