"""
FL Training Data Generator - Multiprocess Version with Joker Filter

Generates training data for FL AI using exhaustive search for 14-card hands.
Reward = Royalties + (FL Stay Bonus: +30 if can stay in FL)

Usage:
    # Generate all hands (mixed joker counts)
    python generate_fl_data.py --samples 10000 --workers 30

    # Generate only hands with 0 jokers (fastest)
    python generate_fl_data.py --samples 5000 --workers 30 --jokers 0

    # Generate only hands with 1 joker
    python generate_fl_data.py --samples 3000 --workers 30 --jokers 1

    # Generate only hands with 2 jokers (slowest)
    python generate_fl_data.py --samples 2000 --workers 30 --jokers 2
"""
import json
import random
import time
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Dict, Any, List, Optional, Tuple
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from game.card import Deck, Card
from ai.fl_solver import solve_fantasyland_exhaustive


FL_STAY_BONUS = 30  # Bonus points for FL stay


def generate_hand_with_joker_count(seed: int, joker_count: int) -> List[Card]:
    """Generate a 14-card hand with exactly the specified number of jokers."""
    random.seed(seed)
    deck = Deck(include_jokers=True)
    deck.shuffle()
    
    # Keep generating until we get a hand with the right joker count
    max_attempts = 1000
    for attempt in range(max_attempts):
        random.seed(seed + attempt * 100000)
        deck = Deck(include_jokers=True)
        deck.shuffle()
        hand = deck.deal(14)
        
        hand_joker_count = sum(1 for c in hand if c.is_joker)
        if hand_joker_count == joker_count:
            return hand
    
    return None


def process_hand_with_filter(args: Tuple[int, Optional[int]]) -> Optional[Dict[str, Any]]:
    """
    Process a single hand with exhaustive search, optionally filtering by joker count.
    
    Args:
        args: Tuple of (seed, joker_filter) where joker_filter is None for any, or 0/1/2
        
    Returns:
        Dictionary with hand data and solution, or None if failed/skipped
    """
    seed, joker_filter = args
    
    try:
        if joker_filter is not None:
            # Generate hand with specific joker count
            hand = generate_hand_with_joker_count(seed, joker_filter)
            if hand is None:
                return None
        else:
            # Generate random hand
            random.seed(seed)
            deck = Deck(include_jokers=True)
            deck.shuffle()
            hand = deck.deal(14)
        
        # Count jokers
        joker_count = sum(1 for c in hand if c.is_joker)
        
        start_time = time.time()
        solutions = solve_fantasyland_exhaustive(hand, max_solutions=1)
        elapsed = time.time() - start_time
        
        if not solutions:
            return None
        
        best = solutions[0]
        
        # Calculate reward: royalties + FL stay bonus
        reward = best.royalties
        if best.can_stay:
            reward += FL_STAY_BONUS
        
        return {
            "seed": seed,
            "joker_count": joker_count,
            "hand": [c.to_dict() for c in hand],
            "solution": {
                "top": [c.to_dict() for c in best.top],
                "middle": [c.to_dict() for c in best.middle],
                "bottom": [c.to_dict() for c in best.bottom],
                "discards": [c.to_dict() for c in best.discards] if best.discards else [],
            },
            "royalties": best.royalties,
            "can_stay": best.can_stay,
            "reward": reward,
            "is_bust": best.is_bust,
            "solve_time": round(elapsed, 2),
        }
    except Exception as e:
        print(f"Error processing seed {seed}: {e}")
        return None


def generate_data(
    num_samples: int,
    output_file: str,
    num_workers: int = 4,
    start_seed: int = 0,
    joker_filter: Optional[int] = None,
):
    """
    Generate training data with multiprocessing.
    
    Args:
        num_samples: Number of samples to generate
        output_file: Path to output JSONL file
        num_workers: Number of parallel workers
        start_seed: Starting seed (for incremental generation)
        joker_filter: None for any, or 0/1/2 for specific joker count
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check existing data for incremental generation
    existing_seeds = set()
    if output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    existing_seeds.add(data['seed'])
                except:
                    pass
        print(f"Found {len(existing_seeds)} existing samples, continuing from there...")
    
    # Generate seeds (skip existing)
    seeds = []
    seed = start_seed
    while len(seeds) < num_samples:
        if seed not in existing_seeds:
            seeds.append(seed)
        seed += 1
    
    # Prepare arguments with joker filter
    task_args = [(s, joker_filter) for s in seeds]
    
    joker_desc = f"jokers={joker_filter}" if joker_filter is not None else "any jokers"
    print(f"Generating {len(seeds)} new samples with {num_workers} workers...")
    print(f"Output: {output_path}")
    print(f"Filter: {joker_desc}")
    print(f"Reward formula: royalties + (FL stay: +{FL_STAY_BONUS})")
    print()
    
    start_time = time.time()
    completed = 0
    
    # Use imap_unordered for real-time progress
    with Pool(num_workers) as pool:
        with open(output_path, 'a', encoding='utf-8') as f:
            # Process results as they complete
            for result in pool.imap_unordered(process_hand_with_filter, task_args):
                if result:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    completed += 1
                    f.flush()
                    
                    # Progress bar
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (len(seeds) - completed) / rate if rate > 0 else 0
                    pct = 100 * completed / len(seeds)
                    bar_len = 30
                    filled = int(bar_len * completed / len(seeds))
                    bar = '█' * filled + '░' * (bar_len - filled)
                    
                    print(f"\r[{bar}] {completed}/{len(seeds)} ({pct:.1f}%) | "
                          f"{rate:.2f}/s | ETA: {eta/60:.1f}min | {joker_desc}", end='', flush=True)
    
    elapsed = time.time() - start_time
    print()
    print(f"\nDone! Generated {completed} samples in {elapsed/60:.1f} minutes")
    print(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate FL training data')
    parser.add_argument('--samples', type=int, default=10000,
                        help='Number of samples to generate (default: 10000)')
    parser.add_argument('--workers', type=int, default=max(1, cpu_count() - 2),
                        help=f'Number of parallel workers (default: {max(1, cpu_count() - 2)})')
    parser.add_argument('--output', type=str, default='ai/data/fl_training_14.jsonl',
                        help='Output file path (default: ai/data/fl_training_14.jsonl)')
    parser.add_argument('--start-seed', type=int, default=0,
                        help='Starting seed (default: 0)')
    parser.add_argument('--jokers', type=int, default=None, choices=[0, 1, 2],
                        help='Filter by joker count (0, 1, or 2). Default: any')
    
    args = parser.parse_args()
    
    generate_data(
        num_samples=args.samples,
        output_file=args.output,
        num_workers=args.workers,
        start_seed=args.start_seed,
        joker_filter=args.jokers,
    )


if __name__ == '__main__':
    main()
