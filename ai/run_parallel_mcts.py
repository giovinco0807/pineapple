import sys
import os
import argparse
import multiprocessing
import subprocess
import time
from pathlib import Path

def run_worker(args):
    worker_id, num_games, out_file = args
    cmd = [
        sys.executable,
        "ai/generate_mcts_supervised.py",
        "--num_games", str(num_games),
        "--output", out_file
    ]
    print(f"Worker {worker_id} starting: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"Worker {worker_id} finished.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_games", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--output", type=str, default="ai/data/mcts_supervised_data_10k.jsonl")
    args = parser.parse_args()

    games_per_worker = args.total_games // args.workers
    remainder = args.total_games % args.workers

    worker_args = []
    tmp_dir = Path("ai/data/tmp_mcts")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.workers):
        n = games_per_worker + (1 if i < remainder else 0)
        out_file = str(tmp_dir / f"worker_{i}.jsonl")
        worker_args.append((i, n, out_file))

    start_time = time.time()
    print(f"Starting {args.workers} workers for a total of {args.total_games} games...")
    
    with multiprocessing.Pool(args.workers) as pool:
        pool.map(run_worker, worker_args)

    print("All workers finished. Merging data...")
    
    # Merge
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as out_f:
        for i in range(args.workers):
            fpath = tmp_dir / f"worker_{i}.jsonl"
            if fpath.exists():
                with open(fpath, 'r', encoding='utf-8') as in_f:
                    for line in in_f:
                        out_f.write(line)
                # Cleanup
                fpath.unlink()
                
    if tmp_dir.exists():
        tmp_dir.rmdir()

    elapsed = time.time() - start_time
    print(f"Finished in {elapsed:.1f}s. Combined data saved to {args.output}")

if __name__ == "__main__":
    main()
