#!/usr/bin/env python3
import os
import subprocess
import time
import shutil

GAMES_PER_ITERATION = 200
ITERATIONS = 500
WORKER_CMD = ["rust_solver/target/release/self_play_worker", str(GAMES_PER_ITERATION)]

def main():
    print("Starting OFC Pineapple Continuous RL Pipeline...")
    
    for i in range(ITERATIONS):
        print(f"\n{'='*40}")
        print(f"ITERATION {i+1}/{ITERATIONS}")
        print(f"{'='*40}")
        
        # 1. Start Inference Server
        print("[1] Starting Inference Server...")
        server_process = subprocess.Popen(["python", "ai/inference_server.py"], env=dict(os.environ, PYTHONPATH="."))
        time.sleep(3) # Wait for server to load models
        
        # 2. Run Self-Play Worker
        print(f"[2] Running Self-Play Worker ({GAMES_PER_ITERATION} games)...")
        # Clear previous data
        if os.path.exists("rust_solver/self_play/self_play_data.jsonl"):
            os.remove("rust_solver/self_play/self_play_data.jsonl")
            
        result = subprocess.run(WORKER_CMD, cwd="rust_solver/self_play", capture_output=True, text=True, check=True)
        print(result.stdout)
        
        # Extract STATS and append to CSV
        for line in result.stdout.split('\n'):
            if line.startswith("STATS|"):
                metrics = line.replace("STATS|", "").strip()
                # Create CSV if not exists
                csv_path = "ai/training_metrics.csv"
                if not os.path.exists(csv_path):
                    with open(csv_path, 'w') as f:
                        f.write("Iteration,P1_Score,P1_Bust,P2_Bust,P1_FL,P2_FL,P1_Royalty,P2_Royalty,P1_QQ,P1_KK,P1_AA,P1_Trips,P2_QQ,P2_KK,P2_AA,P2_Trips\n")
                
                # Check if it has 15 parts, if not we skip or wait for new format
                vals = []
                for part in metrics.split('|'):
                    v = part.split(':')[1].replace('%', '')
                    vals.append(v)
                
                with open(csv_path, 'a') as f:
                    f.write(f"{i+1},{','.join(vals)}\n")
                print(f"[Metric Logged] {metrics}")
        
        # 3. Stop Inference Server before training (free VRAM if necessary, though small model)
        print("[3] Stopping Inference Server...")
        server_process.terminate()
        server_process.wait()
        
        # 4. Train Models
        print("[4] Training Models on new self-play data...")
        models_to_train = ["t0_bb", "t0_btn", "t1_bb", "t1_btn", "t2_bb", "t2_btn", "t3_bb", "t3_btn", "t4_bb", "t4_btn"]
        
        for model_name in models_to_train:
            print(f"  -> Training {model_name}...")
            train_cmd = [
                "python", "ai/train_self_play.py",
                "--model-name", model_name,
                "--output", f"ai/models/t1_placement_net_{model_name}.pt",
                "--epochs", "2"
            ]
            subprocess.run(train_cmd, env=dict(os.environ, PYTHONPATH="."), check=True)
            
        # Auto-plot progress
        print("[5] Plotting updated metrics...")
        subprocess.run(["python", "plot_metrics.py"], env=dict(os.environ, PYTHONPATH="."), check=False)
        
        # Optional: Save a checkpoint copy
        if (i+1) % 10 == 0:
            print("[*] Saving checkpoint...")
            for model_name in ["t1_bb"]:
                src = f"ai/models/t1_placement_net_{model_name}.pt"
                if os.path.exists(src):
                    shutil.copy(src, f"ai/models/checkpoint_{model_name}_iter{i+1}.pt")
                    
        print("[6] Syncing artifacts to GCS...")
        subprocess.run(["gsutil", "-m", "rsync", "-r", "ai/models", "gs://ofc-solver-485418/rl_output/models"], check=False)
        if os.path.exists("ai/training_metrics.csv"):
            subprocess.run(["gsutil", "cp", "ai/training_metrics.csv", "gs://ofc-solver-485418/rl_output/"], check=False)
        if os.path.exists("metrics.png"):
            subprocess.run(["gsutil", "cp", "metrics.png", "gs://ofc-solver-485418/rl_output/"], check=False)
                    
if __name__ == '__main__':
    main()
