import os
import glob
import numpy as np

def merge_t3_datasets(output_file, data_dirs):
    all_files = []
    for d in data_dirs:
        # Recursively find all .npz files in directory
        pattern = os.path.join(d, '**', '*.npz')
        files = glob.glob(pattern, recursive=True)
        print(f"Found {len(files)} NPZ files in {d}")
        all_files.extend(files)
        
    print(f"Total NPZ files to merge: {len(all_files)}")
    
    if not all_files:
        print("No files to merge.")
        return

    # Assuming each NPZ file has states, values, policies, turn, legal_actions, etc.
    # We will just report the count for now, maybe we don't need to physically merge into one huge NPZ file if the DataLoader supports loading from a directory.
    # But usually, it's easier to merge or to just create a manifest.
    # Let's see if we actually need to merge into a single file. A single file with 1M states could be around 1-2 GB, which is fine.
    
    merged_data = {}
    total_states = 0
    
    # We don't want to load all 1M states into memory at once if it's too big, but 1-2GB is totally fine for RAM.
    first = True
    for idx, f in enumerate(all_files):
        try:
            data = np.load(f)
            if first:
                for k in data.keys():
                    merged_data[k] = [data[k]]
                first = False
            else:
                for k in data.keys():
                    merged_data[k].append(data[k])
            
            # Assuming the first dimension of the first array is the number of states
            # e.g. states shape: (N, 14, 4, 13)
            first_key = list(data.keys())[0]
            total_states += len(data[first_key])
            
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx+1}/{len(all_files)} files... ({total_states} states)")
                
        except Exception as e:
            print(f"Error loading {f}: {e}")
            
    print(f"Concatenating arrays for {total_states} states...")
    final_data = {}
    for k, v_list in merged_data.items():
        final_data[k] = np.concatenate(v_list, axis=0)
        
    print(f"Saving to {output_file}...")
    np.savez_compressed(output_file, **final_data)
    print("Done!")

if __name__ == "__main__":
    out_file = "ai/data/t3_dataset_merged.npz"
    dirs_to_merge = [
        r"D:\ofc_data\t3_round2",
        r"C:\Users\Owner\.gemini\antigravity\scratch\ofc-pineapple\ai\data\t3_dataset_round3"
    ]
    merge_t3_datasets(out_file, dirs_to_merge)
