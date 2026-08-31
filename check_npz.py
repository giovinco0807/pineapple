import numpy as np
import glob
import os

d = "/home/Owner/t3_dataset_500k"
files = sorted(glob.glob(os.path.join(d, "*.npz")))
print(f"Total NPZ files: {len(files)}")

if files:
    sample = np.load(files[0])
    keys = list(sample.keys())
    print(f"Keys: {keys}")
    n = sample[keys[0]].shape[0]
    print(f"Samples per file: {n}")
    total = len(files) * n
    print(f"Estimated total samples: {total:,}")
    
    # Latest file
    latest = max(files, key=os.path.getmtime)
    import time
    mtime = os.path.getmtime(latest)
    print(f"Latest file: {os.path.basename(latest)} ({time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))})")
