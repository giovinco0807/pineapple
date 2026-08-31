#!/usr/bin/env python3
"""Inspect the exhaustive T3 data format."""
import json

with open("D:/ofc_data/t3_exhaustive/t3_all.jsonl") as f:
    for i in range(3):
        line = f.readline()
        d = json.loads(line)
        print(f"=== Sample {i} ===")
        print(f"Keys: {list(d.keys())}")
        
        if "choice_records" in d:
            for cr in d["choice_records"][:1]:
                print(f"  Turn: {cr.get('turn')}")
                print(f"  Deal: {cr.get('deal')}")
                print(f"  Top: {cr.get('top')}")
                print(f"  Mid: {cr.get('mid')}")
                print(f"  Bot: {cr.get('bot')}")
                acts = cr.get("actions", [])
                print(f"  Actions ({len(acts)}):")
                for a in acts[:6]:
                    print(f"    {a['desc']}: ev={a['ev']:.1f}")
        elif "board_top" in d:
            print(f"  top: {d['board_top']}")
            print(f"  mid: {d['board_mid']}")
            print(f"  bot: {d['board_bot']}")
            dealt_key = "dealt" if "dealt" in d else "deal"
            print(f"  dealt: {d.get(dealt_key, [])}")
            if "actions" in d:
                acts = d["actions"]
                print(f"  Actions ({len(acts)}):")
                for a in acts[:6]:
                    ev = a.get("ev", a.get("EV"))
                    desc = a.get("desc", a.get("action_desc", ""))
                    print(f"    {desc}: ev={ev}")
        else:
            print(json.dumps(d, indent=2)[:500])
        print()
