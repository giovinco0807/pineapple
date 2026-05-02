import json
import csv
import sys
import os

def convert(jsonl_path, csv_path):
    if not os.path.exists(jsonl_path):
        print(f"File {jsonl_path} does not exist.")
        return

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        data = json.loads(f.readline().strip())

    t0_hand = data.get('t0_hand', '')
    t1_hand = data.get('t1_hand', '')
    placements = data.get('placements', [])

    print(f"Parsed {len(placements)} combinations.")
    
    with open(csv_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Header
        writer.writerow(['T0 Hand', 'T1 Hand', 'T0 Rank', 'T0 Placement', 'Discard', 'T1 Placement', 'EV'])
        
        for p in placements:
            writer.writerow([
                t0_hand,
                t1_hand,
                p.get('t0_idx', ''),
                p.get('t0_p', ''),
                p.get('d', ''),
                p.get('p', ''),
                p.get('ev', 0.0)
            ])
    
    print(f"Exported to {csv_path}")

if __name__ == '__main__':
    convert('test_1hand_full.jsonl', 'test_1hand_full.csv')
