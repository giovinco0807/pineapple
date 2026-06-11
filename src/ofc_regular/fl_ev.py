"""Monte Carlo Fantasyland EV calculator for regular mode."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

from .cards import create_deck
from .fantasyland import solve_fantasyland
from .rules import REGULAR_RULES


@dataclass(frozen=True)
class FLEvSummary:
    trials: int
    solved: int
    avg_royalty: float
    stay_rate: float
    standalone_chain_ev: float
    net_chain_ev: float

    def as_config(self, opponent_avg_royalty: float, line_scoop_advantage: float) -> dict:
        return {
            "rule_set": REGULAR_RULES.name,
            "include_jokers": REGULAR_RULES.include_jokers,
            "deck_cards": 52,
            "fl_entry_cards": dict(REGULAR_RULES.fl_entry_cards),
            "fl_stay_cards": REGULAR_RULES.fl_stay_cards,
            "opponent_avg_royalty": opponent_avg_royalty,
            "line_scoop_advantage": line_scoop_advantage,
            "reward_mode": "chain",
            "fl_stats": {
                "14": {
                    "R": round(self.avg_royalty, 6),
                    "stay_rate": round(self.stay_rate, 6),
                    "count": self.solved,
                }
            },
            "fl_ev": {
                "14": round(self.net_chain_ev, 6),
            },
        }


def estimate_fl_ev(
    *,
    trials: int,
    seed: int,
    stay_bonus: float,
    opponent_avg_royalty: float,
    line_scoop_advantage: float,
) -> FLEvSummary:
    rng = random.Random(seed)
    royalty_sum = 0.0
    stay_count = 0
    solved = 0

    for _ in range(trials):
        deck = create_deck(shuffle=True, rng=rng)
        hand = deck[:14]
        placement = solve_fantasyland(hand, stay_bonus=stay_bonus)
        if placement is None:
            continue
        solved += 1
        royalty_sum += placement.total_royalty
        stay_count += int(placement.can_stay)

    avg_royalty = royalty_sum / max(solved, 1)
    stay_rate = stay_count / max(solved, 1)
    denom = max(1.0 - stay_rate, 1e-9)
    net_immediate = avg_royalty - opponent_avg_royalty + line_scoop_advantage
    return FLEvSummary(
        trials=trials,
        solved=solved,
        avg_royalty=avg_royalty,
        stay_rate=stay_rate,
        standalone_chain_ev=(avg_royalty + line_scoop_advantage) / denom,
        net_chain_ev=net_immediate / denom,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate regular-mode 14-card FL EV")
    parser.add_argument("--trials", type=int, default=100, help="number of random FL hands")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stay-bonus", type=float, default=100.0)
    parser.add_argument("--opponent-avg-royalty", type=float, default=5.0)
    parser.add_argument("--line-scoop-advantage", type=float, default=4.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    summary = estimate_fl_ev(
        trials=args.trials,
        seed=args.seed,
        stay_bonus=args.stay_bonus,
        opponent_avg_royalty=args.opponent_avg_royalty,
        line_scoop_advantage=args.line_scoop_advantage,
    )
    config = summary.as_config(args.opponent_avg_royalty, args.line_scoop_advantage)

    print("Regular FL EV")
    print(f"  trials:       {summary.trials}")
    print(f"  solved:       {summary.solved}")
    print(f"  avg royalty:  {summary.avg_royalty:.3f}")
    print(f"  stay rate:    {summary.stay_rate:.1%}")
    print(f"  line/scoop:   {args.line_scoop_advantage:.3f}")
    print(f"  chain EV:     {summary.standalone_chain_ev:.3f}")
    print(f"  net chain EV: {summary.net_chain_ev:.3f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        print(f"  wrote:        {args.output}")


if __name__ == "__main__":
    main()
