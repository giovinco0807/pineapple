"""
FL EV Calculator — Rust Solver x 100k trials per FL type

Each FL type (QQ/14, KK/15, AA/16, Trips/17) is simulated by:
  1. Dealing N random cards from a 54-card deck (including 2 jokers)
  2. Using the Rust FL solver for optimal placement
  3. Recording royalty (top/mid/bot), can_stay

Usage:
    python ai/calc_fl_ev.py --trials 100000 --workers 8 --seed 42
    python ai/calc_fl_ev.py --trials 1000 --workers 1   # quick test
"""
import sys
import random
import argparse
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai.engine.encoding import ALL_CARDS
from ai.engine.game_engine import RANK_VALUES
from ai.rust_solver_wrapper import RustFLSolver

# ─── Card conversion ─────────────────────────────────────────────────

SUIT_MAP = {'s': 0, 'h': 1, 'd': 2, 'c': 3}


def card_str_to_rust(card_str):
    if card_str.startswith('X'):
        return (0, 4)
    rank = RANK_VALUES.get(card_str[:-1], 0)
    suit = SUIT_MAP.get(card_str[-1], 0)
    return (rank, suit)


# ─── FL types ─────────────────────────────────────────────────────────

FL_TYPES = [
    ('QQ',    14),
    ('KK',    15),
    ('AA',    16),
    ('Trips', 17),
]


# ─── Worker function ──────────────────────────────────────────────────

def worker_fn(args):
    """Run FL trials in a subprocess."""
    fl_type, n_cards, n_trials, seed = args
    rng = random.Random(seed)
    solver = RustFLSolver()

    total_royalty = 0
    total_top_r = 0
    total_mid_r = 0
    total_bot_r = 0
    stay_count = 0
    fail_count = 0
    royalty_list = []

    for _ in range(n_trials):
        hand = rng.sample(ALL_CARDS, n_cards)
        rust_cards = [card_str_to_rust(c) for c in hand]
        result = solver.solve(rust_cards)

        if result is None:
            fail_count += 1
            royalty_list.append(0)
            continue

        top_r = result.get('top_royalty', 0)
        mid_r = result.get('middle_royalty', 0)
        bot_r = result.get('bottom_royalty', 0)
        total = top_r + mid_r + bot_r
        can_stay = result.get('can_stay', False)

        total_royalty += total
        total_top_r += top_r
        total_mid_r += mid_r
        total_bot_r += bot_r
        if can_stay:
            stay_count += 1
        royalty_list.append(total)

    return {
        'fl_type': fl_type,
        'n_cards': n_cards,
        'n_trials': n_trials,
        'total_royalty': total_royalty,
        'total_top_r': total_top_r,
        'total_mid_r': total_mid_r,
        'total_bot_r': total_bot_r,
        'stay_count': stay_count,
        'fail_count': fail_count,
        'royalty_list': royalty_list,
    }


def merge_results(results_list):
    """Merge results from multiple workers for the same FL type."""
    merged = {
        'fl_type': results_list[0]['fl_type'],
        'n_cards': results_list[0]['n_cards'],
        'n_trials': 0,
        'total_royalty': 0,
        'total_top_r': 0,
        'total_mid_r': 0,
        'total_bot_r': 0,
        'stay_count': 0,
        'fail_count': 0,
        'royalty_list': [],
    }
    for r in results_list:
        merged['n_trials'] += r['n_trials']
        merged['total_royalty'] += r['total_royalty']
        merged['total_top_r'] += r['total_top_r']
        merged['total_mid_r'] += r['total_mid_r']
        merged['total_bot_r'] += r['total_bot_r']
        merged['stay_count'] += r['stay_count']
        merged['fail_count'] += r['fail_count']
        merged['royalty_list'].extend(r['royalty_list'])
    return merged


# ─── Display ──────────────────────────────────────────────────────────

def print_results(result):
    n = result['n_trials']
    n_ok = n - result['fail_count']
    avg_r = result['total_royalty'] / n_ok if n_ok > 0 else 0
    avg_top = result['total_top_r'] / n_ok if n_ok > 0 else 0
    avg_mid = result['total_mid_r'] / n_ok if n_ok > 0 else 0
    avg_bot = result['total_bot_r'] / n_ok if n_ok > 0 else 0
    stay_rate = result['stay_count'] / n_ok if n_ok > 0 else 0

    royalties = result['royalty_list']

    print(f"\n--- {result['fl_type']} ({result['n_cards']} cards) ---")
    print(f"  Trials: {n:,} (solver fail: {result['fail_count']})")
    print(f"  Avg Royalty: {avg_r:.2f} (top={avg_top:.2f}, mid={avg_mid:.2f}, bot={avg_bot:.2f})")
    print(f"  Stay Rate: {stay_rate:.1%}")

    # Royalty distribution
    bins = [(0, 0), (1, 5), (6, 10), (11, 15), (16, 20), (21, 30), (31, 50), (51, 999)]
    bin_labels = ['0', '1-5', '6-10', '11-15', '16-20', '21-30', '31-50', '51+']
    print(f"  Royalty Distribution:")
    for (lo, hi), label in zip(bins, bin_labels):
        count = sum(1 for r in royalties if lo <= r <= hi)
        pct = count / n * 100 if n > 0 else 0
        bar = '#' * int(pct / 2)
        print(f"    {label:>5}: {pct:5.1f}% {bar}")

    # Chain EV (standalone = no opponent deduction)
    chain_ev = avg_r / (1 - stay_rate) if stay_rate < 1 else float('inf')
    print(f"  Chain EV (standalone): {chain_ev:.1f}")

    return {
        'type': result['fl_type'],
        'cards': result['n_cards'],
        'avg_royalty': avg_r,
        'stay_rate': stay_rate,
        'chain_ev': chain_ev,
    }


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='FL EV Calculator')
    parser.add_argument('--trials', type=int, default=100000, help='Trials per FL type')
    parser.add_argument('--workers', type=int, default=8, help='Number of parallel workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    print(f"=== FL EV Calculator ===")
    print(f"Trials per type: {args.trials:,}")
    print(f"Workers: {args.workers}")
    print(f"Seed: {args.seed}")

    # Build task list: split each FL type across workers
    tasks = []
    for fl_type, n_cards in FL_TYPES:
        trials_per_worker = args.trials // args.workers
        remainder = args.trials % args.workers
        for w in range(args.workers):
            t = trials_per_worker + (1 if w < remainder else 0)
            worker_seed = args.seed * 1000 + hash((fl_type, w)) % 100000
            tasks.append((fl_type, n_cards, t, worker_seed))

    start = time.time()

    if args.workers == 1:
        # Single-threaded
        raw_results = [worker_fn(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            raw_results = list(pool.map(worker_fn, tasks))

    elapsed = time.time() - start
    print(f"\nCompleted in {elapsed:.1f}s")

    # Group by FL type and merge
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in raw_results:
        grouped[r['fl_type']].append(r)

    print(f"\n{'=' * 60}")
    print(f"=== FL EV Statistics ({args.trials:,} trials each) ===")
    print(f"{'=' * 60}")

    summaries = []
    for fl_type, n_cards in FL_TYPES:
        merged = merge_results(grouped[fl_type])
        s = print_results(merged)
        summaries.append(s)

    # Summary table
    print(f"\n{'=' * 60}")
    print(f"=== Summary ===")
    print(f"{'=' * 60}")
    print(f"  {'Type':<8} {'Cards':>5} {'Avg Roy':>8} {'Stay%':>7} {'Chain EV':>10}")
    print(f"  {'-'*8} {'-'*5} {'-'*8} {'-'*7} {'-'*10}")
    for s in summaries:
        print(f"  {s['type']:<8} {s['cards']:>5} {s['avg_royalty']:>8.2f} {s['stay_rate']:>6.1%} {s['chain_ev']:>10.1f}")

    # Compare with current config
    print(f"\n--- vs current fl_ev.json ---")
    current = {14: {'R': 15.34, 'stay': 0.384},
               15: {'R': 19.30, 'stay': 0.487},
               16: {'R': 23.80, 'stay': 0.641},
               17: {'R': 28.40, 'stay': 0.776}}
    for s in summaries:
        c = current.get(s['cards'], {})
        old_r = c.get('R', 0)
        old_stay = c.get('stay', 0)
        dr = s['avg_royalty'] - old_r
        ds = s['stay_rate'] - old_stay
        print(f"  {s['type']:<8}: R {old_r:.2f} -> {s['avg_royalty']:.2f} ({dr:+.2f}), "
              f"Stay {old_stay:.1%} -> {s['stay_rate']:.1%} ({ds:+.1%})")


if __name__ == '__main__':
    main()
