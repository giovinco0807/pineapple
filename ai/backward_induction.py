"""
Backward Induction Orchestrator

Sends T0 canonical patterns to the Rust backward engine
and collects optimal values + training data.

Usage:
    python -m ai.backward_induction --patterns 1500 --samples 1000 --seed 42
    python -m ai.backward_induction --pattern-file patterns.txt --samples 100
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
import numpy as np

BACKWARD_EXE = os.path.join(
    os.path.dirname(__file__),
    "rust_solver", "target", "release", "backward.exe"
)
PATTERNS_FILE = os.path.join(
    os.path.dirname(__file__),
    "rust_solver", "canonical_patterns.txt"
)
FL_EV_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "config", "fl_ev.json"
)


def load_fl_ev() -> dict:
    """Load FL EV chain values from config."""
    # Default chain EV values
    chain_ev = {"14": 14.0, "15": 27.9, "16": 52.4, "17": 104.5}

    if os.path.exists(FL_EV_CONFIG):
        with open(FL_EV_CONFIG) as f:
            config = json.load(f)
        # Recompute chain EV from stats if available
        stats = config.get("fl_stats", {})
        opp_r = config.get("opponent_avg_royalty", 5.0)
        if stats:
            chain_ev = compute_chain_ev(stats, opp_r)
            print(f"Chain EV from config: {chain_ev}")

    return chain_ev


def compute_chain_ev(stats: dict, opp_r: float) -> dict:
    """
    Compute chain FL EV using recursive formula:
    V(n) = R(n) - opp_R + stay_rate(n) * V(n_next)

    FL entry: QQ->14, KK->15, AA->16, Trips->17
    Stay always goes to 14 cards (QQ equivalent).
    """
    # Sort by card count descending for recursive computation
    card_counts = sorted(stats.keys(), key=int, reverse=True)

    # V(14) base case: no chain from QQ stay (stays go to 14 again)
    # V(n) = R(n) - opp_R + stay_rate(n) * V(14)
    # V(14) = R(14) - opp_R + stay_rate(14) * V(14)
    # V(14) = (R(14) - opp_R) / (1 - stay_rate(14))
    r14 = stats["14"]["R"]
    sr14 = stats["14"]["stay_rate"]
    if sr14 >= 1.0:
        sr14 = 0.99  # safety
    v14 = (r14 - opp_r) / (1.0 - sr14)

    chain = {"14": round(v14, 1)}
    for n in card_counts:
        if n == "14":
            continue
        rn = stats[n]["R"]
        srn = stats[n]["stay_rate"]
        vn = rn - opp_r + srn * v14
        chain[n] = round(vn, 1)

    return chain


def load_patterns(pattern_file: str = None, n_patterns: int = None,
                  seed: int = 42) -> list:
    """Load canonical T0 patterns, optionally sampling a subset."""
    src = pattern_file or PATTERNS_FILE
    if not os.path.exists(src):
        raise FileNotFoundError(f"Pattern file not found: {src}")

    with open(src) as f:
        patterns = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(patterns)} canonical patterns from {src}")

    if n_patterns and n_patterns < len(patterns):
        rng = random.Random(seed)
        patterns = rng.sample(patterns, n_patterns)
        print(f"Sampled {n_patterns} patterns (seed={seed})")

    return patterns


def run_backward(patterns: list, n_samples: int, fl_ev: dict,
                 seed: int = 42, bust_penalty: float = -6.0,
                 output_file: str = None):
    """
    Run backward engine on a list of T0 patterns.
    Sends JSON requests via stdin, reads responses from stdout.
    """
    if not os.path.exists(BACKWARD_EXE):
        raise FileNotFoundError(f"Backward exe not found: {BACKWARD_EXE}")

    proc = subprocess.Popen(
        [BACKWARD_EXE, "stdin"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    results = []
    total = len(patterns)
    start_time = time.time()

    try:
        for i, pattern_str in enumerate(patterns):
            cards = [c.strip() for c in pattern_str.split(",")]
            req = {
                "t0_hand": cards,
                "n_samples": n_samples,
                "seed": seed + i,  # Different seed per pattern
                "fl_ev": fl_ev,
                "bust_penalty": bust_penalty,
            }

            proc.stdin.write(json.dumps(req) + "\n")
            proc.stdin.flush()

            resp_line = proc.stdout.readline()
            if not resp_line:
                print(f"ERROR: No response for pattern {i}")
                break

            resp = json.loads(resp_line.strip())
            if "error" in resp:
                print(f"ERROR for {pattern_str}: {resp['error']}")
                continue

            results.append({
                "pattern": pattern_str,
                "avg_value": resp["avg_value"],
                "min_value": resp["min_value"],
                "max_value": resp["max_value"],
                "elapsed_ms": resp["elapsed_ms"],
            })

            # Progress
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * (total - i - 1)
            print(f"[{i+1}/{total}] {pattern_str}: "
                  f"avg={resp['avg_value']:.1f} "
                  f"({resp['elapsed_ms']/1000:.1f}s) "
                  f"ETA: {eta/60:.0f}min", flush=True)

    finally:
        proc.stdin.close()
        proc.wait()

    total_time = time.time() - start_time
    print(f"\nCompleted {len(results)}/{total} patterns in {total_time:.0f}s")

    # Save results
    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")

    # Summary stats
    if results:
        values = [r["avg_value"] for r in results]
        print(f"\nSummary:")
        print(f"  Mean value: {np.mean(values):.2f}")
        print(f"  Std value: {np.std(values):.2f}")
        print(f"  Min: {min(values):.2f}")
        print(f"  Max: {max(values):.2f}")
        print(f"  Avg time/pattern: {np.mean([r['elapsed_ms'] for r in results])/1000:.1f}s")

    return results


def main():
    parser = argparse.ArgumentParser(description="Backward Induction Orchestrator")
    parser.add_argument("--patterns", type=int, default=10,
                        help="Number of random patterns to solve (default: 10)")
    parser.add_argument("--pattern-file", type=str, default=None,
                        help="File with specific patterns (one per line)")
    parser.add_argument("--samples", type=int, default=100,
                        help="Samples per pattern (default: 100)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="data/backward_results.json",
                        help="Output file")
    parser.add_argument("--bust-penalty", type=float, default=-6.0)
    args = parser.parse_args()

    print("=== Backward Induction Orchestrator ===")
    print(f"Backward exe: {BACKWARD_EXE}")

    # Load FL EV
    fl_ev = load_fl_ev()
    print(f"FL EV: {fl_ev}")

    # Load patterns
    if args.pattern_file:
        patterns = load_patterns(pattern_file=args.pattern_file)
    else:
        patterns = load_patterns(n_patterns=args.patterns, seed=args.seed)

    # Run
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    results = run_backward(
        patterns, args.samples, fl_ev,
        seed=args.seed, bust_penalty=args.bust_penalty,
        output_file=args.output,
    )

    return results


if __name__ == "__main__":
    main()
