import json

lines = open('t0_filtered_train.jsonl', encoding='utf-8').readlines()
# Show last 4 joker hands
for line in lines[-4:]:
    d = json.loads(line)
    hand = d['hand']
    n_placements = len(d.get('placements', []))
    placements = d.get('placements', [])
    
    print(f"\n{'='*60}")
    print(f"Hand: {hand}")
    print(f"Total placements evaluated: {n_placements}")
    
    if placements:
        # Sort by EV descending
        sorted_p = sorted(placements, key=lambda x: x['ev'], reverse=True)
        print(f"\nTop 5 placements:")
        for i, p in enumerate(sorted_p[:5]):
            print(f"  #{i+1}: {p['p']} -> EV: {p['ev']:+.2f}")
        print(f"\nWorst placement:")
        worst = sorted_p[-1]
        print(f"  #{n_placements}: {worst['p']} -> EV: {worst['ev']:+.2f}")
        print(f"\nEV range: {sorted_p[-1]['ev']:+.2f} to {sorted_p[0]['ev']:+.2f}")
