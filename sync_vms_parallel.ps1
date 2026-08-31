$vms = @("t3-dataset-gen-2", "t3-dataset-gen-3", "t3-dataset-gen-4", "t3-dataset-gen-5", "t3-dataset-gen-6")

$jobs = @()

foreach ($vm in $vms) {
    $scriptBlock = {
        param($targetVm)
        Write-Host "Starting on $targetVm"
        gcloud compute ssh $targetVm --zone us-central1-a --quiet --command "pkill -f generate_t3_data.py; mkdir -p ofc-pineapple/ai"
        gcloud compute scp ai\generate_t3_data.py ${targetVm}:ofc-pineapple/ai/generate_t3_data.py --zone us-central1-a --quiet
        gcloud compute ssh $targetVm --zone us-central1-a --quiet --command "cd ofc-pineapple && nohup ./venv/bin/python3 ai/generate_t3_data.py --states 500000 --workers 30 --save-interval 1000 --output-dir /home/Owner/t3_dataset_round3 > /home/Owner/t3_gen_round3.log 2>&1 &"
        Write-Host "Finished $targetVm"
    }
    $jobs += Start-Job -ScriptBlock $scriptBlock -ArgumentList $vm
}

Wait-Job -Job $jobs
Receive-Job -Job $jobs
