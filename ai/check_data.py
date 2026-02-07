import json
with open('ai/data/fl_training_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

stats = data['stats']
print('=== Training Data Stats ===')
print(f'Samples: {stats["num_samples"]}')
print(f'Cards per hand: {stats["cards_per_hand"]}')
print(f'Include jokers: {stats["include_jokers"]}')
print(f'Avg time per hand: {stats["avg_time_per_hand"]:.2f}s')
print(f'Stay rate: {stats["stay_rate"]*100:.1f}%')
print(f'Avg royalties: {stats["avg_royalties"]:.1f}')
print(f'Bust-only hands: {stats["hands_with_bust_only"]}')

print()
print('=== Sample 0 ===')
s = data['data'][0]
print(f'Hand: {s["hand_str"]}')
print(f'Has joker: {s["has_joker"]}')
if s['optimal']:
    opt = s['optimal_str']
    print(f'Top:    {opt["top"]}')
    print(f'Middle: {opt["middle"]}')
    print(f'Bottom: {opt["bottom"]}')
    print(f'Discard: {opt["discards"]}')
    print(f'Royalties: {s["optimal"]["royalties"]}')
    print(f'Can stay: {s["optimal"]["can_stay"]}')
