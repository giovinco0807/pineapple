"""Analyze selfplay JSONL: compute actual FL_EV and bust_penalty.

Usage:
    python ai/training/analyze_selfplay.py ai/rust_benchmark/results/all_selfplay.jsonl
"""
import json
import sys
import statistics
from collections import defaultdict

def load_games(path):
    games = []
    for line in open(path):
        try:
            games.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return games

def infer_process_id(seed):
    """Infer which process a game belongs to from its seed.

    Process layout:
    VM 0: base 50000, procs at +0*250, +1*250, ..., +11*250
    VM 1: base 53000, procs at +0*250, ..., +11*250
    VM 2: base 56000, ...
    VM 3: base 59000, ...
    VM 4: base 62000, ...

    Each proc has 250 games with seeds: base + proc*250 + 1 ... base + proc*250 + 250
    """
    for vm_base in [50000, 53000, 56000, 59000, 62000]:
        for proc in range(12):
            proc_base = vm_base + proc * 250
            if proc_base < seed <= proc_base + 250:
                return proc_base
    return None

def reconstruct_fl_chains(games):
    """Reconstruct FL chains by processing games sequentially within each process."""
    # Group by process
    by_process = defaultdict(list)
    for g in games:
        pid = infer_process_id(g['seed'])
        if pid is not None:
            by_process[pid].append(g)

    # Sort within each process by seed (= sequential order)
    for pid in by_process:
        by_process[pid].sort(key=lambda g: g['seed'])

    # Track FL chains
    fl_chains = []  # list of (fl_card_count, [normal_scores_in_chain])

    for pid, proc_games in sorted(by_process.items()):
        hero_fl_active = False
        opp_fl_active = False
        hero_fl_cards = 0
        opp_fl_cards = 0
        current_chain_hero = None  # (fl_cards, [scores])
        current_chain_opp = None
        prev_session = None

        for g in proc_games:
            res = g['result']
            sid = g['session_id']

            # Session boundary → reset FL state
            if prev_session is not None and sid != prev_session:
                if current_chain_hero:
                    fl_chains.append(current_chain_hero)
                    current_chain_hero = None
                if current_chain_opp:
                    fl_chains.append(current_chain_opp)
                    current_chain_opp = None
                hero_fl_active = False
                opp_fl_active = False
                hero_fl_cards = 0
                opp_fl_cards = 0
            prev_session = sid

            # New format: use is_fl_hand field if available
            if 'is_fl_hand' in g:
                is_fl_hand = g['is_fl_hand']
            else:
                is_fl_hand = hero_fl_active or opp_fl_active

            # FL stay detection:
            #   New format: use hero_fl_stay / opp_fl_stay fields directly
            #   Old format fallback: hero_fl_type == "trips" (misses Bot Quads+)
            def hero_fl_stay_check(res):
                if res['hero_busted']:
                    return False
                # New JSONL format has hero_fl_stay field
                if 'hero_fl_stay' in res:
                    return res['hero_fl_stay']
                # Old format: approximate with hero_fl_type == "trips"
                ft = res.get('hero_fl_type')
                return ft == "trips"

            def opp_fl_stay_check(res):
                if res['opp_busted']:
                    return False
                # New JSONL format has opp_fl_stay field
                if 'opp_fl_stay' in res:
                    return res['opp_fl_stay']
                # Old format: use opp_fl_cards == 17 as proxy for Trips
                return res.get('opp_fl') and res.get('opp_fl_cards') == 17

            if is_fl_hand:
                # This game is an FL hand
                if hero_fl_active and current_chain_hero:
                    current_chain_hero[1].append(res['normal_score'])
                if opp_fl_active and current_chain_opp:
                    current_chain_opp[1].append(-res['normal_score'])  # opp perspective

                # Check if FL continues
                if hero_fl_active:
                    if hero_fl_stay_check(res):
                        pass  # hero stays in FL (Top Trips)
                    else:
                        # Chain ends
                        if current_chain_hero:
                            fl_chains.append(current_chain_hero)
                            current_chain_hero = None
                        hero_fl_active = False
                        hero_fl_cards = 0

                if opp_fl_active:
                    if opp_fl_stay_check(res):
                        pass  # opp stays
                    else:
                        if current_chain_opp:
                            fl_chains.append(current_chain_opp)
                            current_chain_opp = None
                        opp_fl_active = False
                        opp_fl_cards = 0

                # Check new FL entries from the non-FL player
                if not hero_fl_active and not res['hero_busted'] and res['hero_fl']:
                    hero_fl_active = True
                    hero_fl_cards = res['hero_fl_cards']
                    current_chain_hero = (hero_fl_cards, [])
                if not opp_fl_active and not res['opp_busted'] and res['opp_fl']:
                    opp_fl_active = True
                    opp_fl_cards = res['opp_fl_cards']
                    current_chain_opp = (opp_fl_cards, [])
            else:
                # Normal hand - check FL entry
                if not res['hero_busted'] and res['hero_fl']:
                    hero_fl_active = True
                    hero_fl_cards = res['hero_fl_cards']
                    current_chain_hero = (hero_fl_cards, [])

                if not res['opp_busted'] and res['opp_fl']:
                    opp_fl_active = True
                    opp_fl_cards = res['opp_fl_cards']
                    current_chain_opp = (opp_fl_cards, [])

        # End of process: close any open chains
        if current_chain_hero:
            fl_chains.append(current_chain_hero)
        if current_chain_opp:
            fl_chains.append(current_chain_opp)

    return fl_chains

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'ai/rust_benchmark/results/all_selfplay.jsonl'
    games = load_games(path)
    print(f"Loaded {len(games)} games")

    # ============================================================
    # 1. bust_penalty
    # ============================================================
    normal_scores_no_bust = [g['result']['normal_score'] for g in games if not g['result']['hero_busted']]
    normal_scores_bust = [g['result']['normal_score'] for g in games if g['result']['hero_busted']]

    avg_no_bust = statistics.mean(normal_scores_no_bust) if normal_scores_no_bust else 0
    avg_bust = statistics.mean(normal_scores_bust) if normal_scores_bust else 0
    bust_penalty = avg_no_bust - avg_bust

    print(f"\n{'='*60}")
    print(f"BUST PENALTY")
    print(f"{'='*60}")
    print(f"  E[normal_score | no bust] = {avg_no_bust:+.2f}  (n={len(normal_scores_no_bust)})")
    print(f"  E[normal_score | bust]    = {avg_bust:+.2f}  (n={len(normal_scores_bust)})")
    print(f"  bust_penalty = {bust_penalty:.2f}")
    print(f"  (Recommended --bust-penalty {bust_penalty:.1f})")

    # ============================================================
    # 2. FL_EV from chain reconstruction
    # ============================================================
    fl_chains = reconstruct_fl_chains(games)

    print(f"\n{'='*60}")
    print(f"FL_EV (actual chain values)")
    print(f"{'='*60}")
    print(f"  Total FL chains reconstructed: {len(fl_chains)}")

    # Group by FL card count
    chains_by_cards = defaultdict(list)
    for fc, scores in fl_chains:
        chain_value = sum(scores)
        chain_len = len(scores)
        chains_by_cards[fc].append((chain_value, chain_len))

    fl_ev_actual = {}
    card_labels = {14: "QQ", 15: "KK", 16: "AA", 17: "Trips"}
    for fc in sorted(chains_by_cards.keys()):
        values = [v for v, l in chains_by_cards[fc]]
        lengths = [l for v, l in chains_by_cards[fc]]
        n = len(values)
        avg = statistics.mean(values)
        std = statistics.stdev(values) if n > 1 else 0
        avg_len = statistics.mean(lengths)
        fl_ev_actual[fc] = avg
        label = card_labels.get(fc, f"{fc}cards")
        print(f"  {label} ({fc} cards): EV={avg:+.1f} +/- {std:.1f}  chain_len={avg_len:.1f}  n={n}")

    # Current config FL_EV for comparison
    print(f"\n  Current config FL_EV:")
    config_ev = {14: 14.0, 15: 27.9, 16: 52.4, 17: 104.5}
    for fc in sorted(config_ev.keys()):
        label = card_labels.get(fc, f"{fc}")
        actual = fl_ev_actual.get(fc, 0)
        diff = actual - config_ev[fc]
        print(f"    {label}: config={config_ev[fc]:.1f}  actual={actual:+.1f}  diff={diff:+.1f}")

    # ============================================================
    # 3. Summary stats
    # ============================================================
    # Separate by is_fl_hand if available
    has_fl_field = any('is_fl_hand' in g for g in games)
    if has_fl_field:
        normal_games = [g for g in games if not g.get('is_fl_hand', False)]
        fl_games = [g for g in games if g.get('is_fl_hand', False)]
    else:
        normal_games = games
        fl_games = []

    n_normal = len(normal_games)
    n_fl = len(fl_games)

    hero_busts = sum(1 for g in normal_games if g['result']['hero_busted'])
    opp_busts = sum(1 for g in normal_games if g['result']['opp_busted'])
    hero_fl_entries = sum(1 for g in normal_games if g['result']['hero_fl'] and not g['result']['hero_busted'])
    opp_fl_entries = sum(1 for g in normal_games if g['result']['opp_fl'] and not g['result']['opp_busted'])

    # FL stay stats
    hero_fl_stays = sum(1 for g in fl_games if g['result'].get('hero_fl_stay', False))
    opp_fl_stays = sum(1 for g in fl_games if g['result'].get('opp_fl_stay', False))
    hero_in_fl = sum(1 for g in fl_games if g.get('fl_active', [False, False])[0])
    opp_in_fl = sum(1 for g in fl_games if g.get('fl_active', [False, False])[1])

    total_scores = [g['result']['total_score'] for g in games]
    normal_scores = [g['result']['normal_score'] for g in games]

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Total games: {len(games)} (normal={n_normal}, FL={n_fl})")
    print(f"  --- Normal Hands ---")
    if n_normal > 0:
        print(f"  Hero bust: {hero_busts}/{n_normal} ({100*hero_busts/n_normal:.1f}%)")
        print(f"  Opp bust:  {opp_busts}/{n_normal} ({100*opp_busts/n_normal:.1f}%)")
        print(f"  Hero FL entry: {hero_fl_entries}/{n_normal} ({100*hero_fl_entries/n_normal:.1f}%)")
        print(f"  Opp FL entry:  {opp_fl_entries}/{n_normal} ({100*opp_fl_entries/n_normal:.1f}%)")
    if n_fl > 0:
        print(f"  --- FL Hands ---")
        print(f"  Hero in FL: {hero_in_fl}, stay: {hero_fl_stays} ({100*hero_fl_stays/max(hero_in_fl,1):.0f}%)")
        print(f"  Opp in FL:  {opp_in_fl}, stay: {opp_fl_stays} ({100*opp_fl_stays/max(opp_in_fl,1):.0f}%)")
    print(f"  --- Scores ---")
    print(f"  Avg total_score: {statistics.mean(total_scores):+.2f}")
    print(f"  Avg normal_score: {statistics.mean(normal_scores):+.2f}")

    # Output recommended config
    print(f"\n{'='*60}")
    print(f"RECOMMENDED CONFIG")
    print(f"{'='*60}")
    print(f"  --bust-penalty {bust_penalty:.1f}")
    print(f"  fl_ev.json:")
    fl_ev_out = {}
    for fc in [14, 15, 16, 17]:
        val = fl_ev_actual.get(fc, config_ev.get(fc, 0))
        fl_ev_out[fc] = round(val, 1)
        label = card_labels.get(fc, str(fc))
        print(f"    {fc}: {val:.1f}  ({label})")

    # Write updated fl_ev.json
    import os
    config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config')
    os.makedirs(config_dir, exist_ok=True)

    fl_ev_path = os.path.join(config_dir, 'fl_ev.json')
    # Read existing
    try:
        with open(fl_ev_path) as f:
            fl_config = json.load(f)
    except:
        fl_config = {}

    # Update values
    fl_config['fl_ev'] = {str(k): v for k, v in fl_ev_out.items()}
    fl_config['source'] = 'selfplay_2372games'
    fl_config['bust_penalty'] = round(bust_penalty, 1)

    with open(fl_ev_path, 'w') as f:
        json.dump(fl_config, f, indent=2)
    print(f"\n  Written to {fl_ev_path}")

if __name__ == '__main__':
    main()
