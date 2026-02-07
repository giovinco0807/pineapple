"""
FL Training Data Analysis

Analyzes the generated FL training data to understand patterns
and prepare for model training.

Usage:
    python analyze_data.py --input ai/data/fl_training_14.jsonl
"""
import json
import argparse
from pathlib import Path
from collections import Counter
from typing import Dict, Any, List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_data(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL data file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                pass
    return data


def analyze_rewards(data: List[Dict[str, Any]]):
    """Analyze reward distribution."""
    rewards = [d['reward'] for d in data]
    royalties = [d['royalties'] for d in data]
    can_stay = [d['can_stay'] for d in data]
    
    print("=" * 60)
    print("REWARD ANALYSIS")
    print("=" * 60)
    print(f"Total samples: {len(data)}")
    print()
    
    print("Reward distribution:")
    print(f"  Min: {min(rewards)}")
    print(f"  Max: {max(rewards)}")
    print(f"  Mean: {sum(rewards)/len(rewards):.2f}")
    print()
    
    print("Royalty distribution:")
    print(f"  Min: {min(royalties)}")
    print(f"  Max: {max(royalties)}")
    print(f"  Mean: {sum(royalties)/len(royalties):.2f}")
    print()
    
    stay_count = sum(1 for s in can_stay if s)
    print(f"FL Stay rate: {stay_count}/{len(data)} ({100*stay_count/len(data):.1f}%)")
    print()


def analyze_hand_patterns(data: List[Dict[str, Any]]):
    """Analyze common hand patterns in solutions."""
    print("=" * 60)
    print("HAND PATTERN ANALYSIS")
    print("=" * 60)
    
    # Count card usage in each row
    top_patterns = Counter()
    middle_patterns = Counter()
    bottom_patterns = Counter()
    
    for d in data:
        sol = d['solution']
        
        # Top: count pairs/trips
        top_ranks = [c['r'] for c in sol['top']]
        if len(set(top_ranks)) == 1:
            top_patterns['Trips'] += 1
        elif len(set(top_ranks)) == 2:
            top_patterns['Pair'] += 1
        else:
            top_patterns['High Card'] += 1
    
    print("Top row patterns:")
    for pattern, count in top_patterns.most_common():
        print(f"  {pattern}: {count} ({100*count/len(data):.1f}%)")
    print()


def analyze_solve_times(data: List[Dict[str, Any]]):
    """Analyze solve time distribution."""
    print("=" * 60)
    print("SOLVE TIME ANALYSIS")
    print("=" * 60)
    
    times = [d.get('solve_time', 0) for d in data]
    if times:
        print(f"Min: {min(times):.1f}s")
        print(f"Max: {max(times):.1f}s")
        print(f"Mean: {sum(times)/len(times):.1f}s")
        print(f"Total: {sum(times)/3600:.1f} hours")
    print()


def main():
    parser = argparse.ArgumentParser(description='Analyze FL training data')
    parser.add_argument('--input', type=str, default='ai/data/fl_training_14.jsonl',
                        help='Input JSONL file')
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"Error: File not found: {args.input}")
        return
    
    data = load_data(args.input)
    
    if not data:
        print("No data found!")
        return
    
    analyze_rewards(data)
    analyze_hand_patterns(data)
    analyze_solve_times(data)


if __name__ == '__main__':
    main()
