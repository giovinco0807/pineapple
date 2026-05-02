"""Count total hands across all GCS worker files."""
import subprocess

result = subprocess.run(
    ['gsutil', 'cat', 'gs://ofc-solver-485418/t0_phase_e/worker_*.jsonl'],
    capture_output=True, text=True, encoding='utf-8'
)

lines = [l for l in result.stdout.strip().split('\n') if l.strip()]
print(f"Total GCP hands: {len(lines)}")

# Count per worker
import re
from collections import Counter
result2 = subprocess.run(
    ['gsutil', 'ls', '-l', 'gs://ofc-solver-485418/t0_phase_e/worker_*.jsonl'],
    capture_output=True, text=True
)
worker_files = [l for l in result2.stdout.strip().split('\n') if 'worker_' in l]
print(f"Worker files: {len(worker_files)}")

# Local progress
import os
local_path = r'c:\Users\Owner\.gemini\antigravity\scratch\ofc-pineapple\t0_local_worker.jsonl'
if os.path.exists(local_path):
    with open(local_path, encoding='utf-8') as f:
        local_lines = f.readlines()
    print(f"Local hands: {len(local_lines)}")
    print(f"Local file size: {os.path.getsize(local_path) / 1024:.0f} KB")
else:
    print("Local worker file not found")

print(f"\nGRAND TOTAL: {len(lines) + len(local_lines)} hands")
print(f"Target: 3100 hands")
print(f"Progress: {(len(lines) + len(local_lines)) / 3100 * 100:.1f}%")
