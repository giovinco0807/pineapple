import subprocess
import json
import concurrent.futures

def stop_instance(name, zone):
    print(f"Stopping {name} in {zone}...")
    subprocess.run(["gcloud.cmd", "compute", "instances", "stop", name, "--zone", zone, "--quiet"], capture_output=True)
    print(f"Stopped {name}")

output = subprocess.run(["gcloud.cmd", "compute", "instances", "list", "--filter=name~'^t0-e-.*'", "--format=json"], capture_output=True)
instances = json.loads(output.stdout)

with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    for instance in instances:
        if instance['status'] != 'TERMINATED':
            executor.submit(stop_instance, instance['name'], instance['zone'].split('/')[-1])
