#!/usr/bin/env python3
"""
Convert T1 MC data (from generate_t1_mc.py) to training format (for train_t1_placement.py).

Input format (per line):
  {
    "t0_hand": "Tc 9s Ad 3s Ah",
    "t1_hand": "X2 8d 5c",
    "placements": [
      {"t0_idx": 4, "t0_p": "Tc->Bottom, 9s->Middle, Ad->Top, 3s->Bottom, Ah->Top",
       "d": "5c", "p": "X2\u2192Bottom, 8d\u2192Middle", "ev": 44.056},
      ...
    ]
  }

Output format (per line):
  {
    "board": "Top[Ad Ah] Mid[9s] Bot[Tc 3s]",
    "hand": "X2 8d 5c",
    "placements": [{"d": "5c", "p": "X2\u2192Bottom, 8d\u2192Middle", "ev": 44.056}, ...],
    "original_ev": 44.056
  }

Note: Since each input record may have placements from multiple T0 arrangements,
we group placements by t0_idx and produce one training sample per unique T0 board.
"""

import json
import argparse
from collections import defaultdict


def t0_p_to_board(t0_p_str):
    """Convert T0 placement string to board string."""
    top, mid, bot = [], [], []
    for part in t0_p_str.split(", "):
        card, row = part.split("->")
        if row == "Top":
            top.append(card)
        elif row == "Middle":
            mid.append(card)
        elif row == "Bottom":
            bot.append(card)
    return "Top[{}] Mid[{}] Bot[{}]".format(" ".join(top), " ".join(mid), " ".join(bot))


def convert_record(record):
    """Convert one MC record to training samples."""
    t1_hand = record["t1_hand"]
    placements = record.get("placements", [])
    
    if not placements:
        return []
    
    # Group by t0_idx to create separate training samples
    groups = defaultdict(list)
    t0_boards = {}
    
    for p in placements:
        t0_idx = p.get("t0_idx", 0)
        groups[t0_idx].append({
            "d": p["d"],
            "p": p["p"],
            "ev": p["ev"],
        })
        if t0_idx not in t0_boards:
            t0_boards[t0_idx] = t0_p_to_board(p["t0_p"])
    
    results = []
    for t0_idx, plist in groups.items():
        board = t0_boards[t0_idx]
        plist.sort(key=lambda x: x["ev"], reverse=True)
        best_ev = plist[0]["ev"]
        results.append({
            "board": board,
            "hand": t1_hand,
            "placements": plist,
            "original_ev": best_ev,
        })
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Convert T1 MC data to training format")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--best-t0-only", action="store_true",
                        help="Only keep the best T0 placement per hand (highest EV)")
    args = parser.parse_args()
    
    total_in = 0
    total_out = 0
    
    with open(args.input, encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            total_in += 1
            
            samples = convert_record(record)
            
            if args.best_t0_only and samples:
                # Keep only the sample with the best EV
                samples.sort(key=lambda x: x["original_ev"], reverse=True)
                samples = samples[:1]
            
            for s in samples:
                fout.write(json.dumps(s, ensure_ascii=False) + "\n")
                total_out += 1
    
    print(f"Converted {total_in} MC records -> {total_out} training samples")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
