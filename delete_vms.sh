gcloud compute instances list --filter="name~t1-mc" --format="csv(name,zone)" | tail -n +2 > vms_to_delete.txt
while IFS=, read -r name zone; do
    echo "Deleting $name in $zone"
    gcloud compute instances delete "$name" --zone="$zone" --quiet &
done < vms_to_delete.txt
wait
echo "All VMs deleted"
