$vms = @("t3-dataset-gen-2", "t3-dataset-gen-3", "t3-dataset-gen-4", "t3-dataset-gen-5", "t3-dataset-gen-6")

foreach ($vm in $vms) {
    Write-Host "Processing $vm ..."
    # Kill old processes
    gcloud compute ssh $vm --zone us-central1-a --command "pkill -f generate_t3_data.py"
    
    # Create the ai directory just in case it's missing on gen-6
    gcloud compute ssh $vm --zone us-central1-a --command "mkdir -p ofc-pineapple/ai"

    # SCP the local file
    gcloud compute scp ai\generate_t3_data.py ${vm}:ofc-pineapple/ai/generate_t3_data.py --zone us-central1-a
    
    # Start the new process
    gcloud compute ssh $vm --zone us-central1-a --command "cd ofc-pineapple && nohup ./venv/bin/python3 ai/generate_t3_data.py --states 500000 --workers 30 --save-interval 1000 --output-dir /home/Owner/t3_dataset_round3 > /home/Owner/t3_gen_round3.log 2>&1 &"
}
