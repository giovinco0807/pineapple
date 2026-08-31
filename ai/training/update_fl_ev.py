"""
Update FL_EV values from self-play game logs.

Reads JSONL from Rust self-play, aggregates FL chain results by FL type,
and updates ai/config/fl_ev.json.

Usage:
    python ai/training/update_fl_ev.py data/selfplay.jsonl
    python ai/training/update_fl_ev.py data/sp_iter1.jsonl data/sp_iter2.jsonl
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict

CONFIG_PATH = Path(__file__).parent.parent / "config" / "fl_ev.json"

# Default FL_EV (chain formula)
DEFAULT_FL_EV = {
    "14": 14.0,   # QQ
    "15": 27.9,   # KK
    "16": 52.4,   # AA
    "17": 104.5,  # Trips
}

FL_NAMES = {14: "QQ", 15: "KK", 16: "AA", 17: "Trips"}


def aggregate_fl_ev(jsonl_paths):
    """Aggregate FL chain EV from self-play JSONL files."""
    fl_scores = defaultdict(list)  # fl_cards -> list of total_scores

    total_games = 0
    total_fl = 0

    for path in jsonl_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    game = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                result = game["result"]
                total_games += 1

                # Hero FL
                if result["hero_fl"]:
                    fl_cards = result["hero_fl_cards"]
                    if fl_cards > 0:
                        fl_scores[fl_cards].append(result["total_score"])
                        total_fl += 1

                # Opp FL (from opp's perspective: score = -hero_score)
                if result["opp_fl"]:
                    fl_cards = result["opp_fl_cards"]
                    if fl_cards > 0:
                        fl_scores[fl_cards].append(-result["total_score"])
                        total_fl += 1

    return fl_scores, total_games, total_fl


def compute_fl_ev(fl_scores, min_samples=20):
    """Compute FL_EV from aggregated scores.

    FL_EV[cards] = mean score when entering FL with that hand type.
    Only update if enough samples; keep default otherwise.
    """
    fl_ev = dict(DEFAULT_FL_EV)

    for fl_cards, scores in sorted(fl_scores.items()):
        name = FL_NAMES.get(fl_cards, f"?{fl_cards}")
        n = len(scores)
        mean = sum(scores) / n if n > 0 else 0
        std = (sum((s - mean) ** 2 for s in scores) / max(n - 1, 1)) ** 0.5

        print(f"  {name:5s} (cards={fl_cards}): n={n:4d}, mean={mean:+.1f}, std={std:.1f}")

        if n >= min_samples:
            fl_ev[str(fl_cards)] = round(mean, 1)
        else:
            print(f"    -> keeping default ({fl_ev.get(str(fl_cards), 'N/A')}), need {min_samples} samples")

    return fl_ev


def main():
    parser = argparse.ArgumentParser(description="Update FL_EV from self-play logs")
    parser.add_argument("inputs", nargs="+", help="Input JSONL files")
    parser.add_argument("--min-samples", type=int, default=20,
                        help="Minimum samples to update FL_EV (default: 20)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print results without saving")
    args = parser.parse_args()

    print("Aggregating FL results...")
    fl_scores, total_games, total_fl = aggregate_fl_ev(args.inputs)
    print(f"  {total_games} games, {total_fl} FL entries ({total_fl/max(total_games,1)*100:.1f}%)")
    print()

    fl_ev = compute_fl_ev(fl_scores, args.min_samples)
    print()
    print(f"Updated FL_EV: {json.dumps(fl_ev, indent=2)}")

    if not args.dry_run:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Load existing config
        config = {}
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "r") as f:
                config = json.load(f)
        config["fl_ev"] = fl_ev
        with open(CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=2)
        print(f"\nSaved to {CONFIG_PATH}")
    else:
        print("\n(dry-run, not saved)")


if __name__ == "__main__":
    main()
