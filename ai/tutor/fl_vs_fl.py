"""Fantasyland-vs-Fantasyland expectation by direct convolution.

When both players are in FL neither makes a decision: both boards are set
blind at deal time, so the matchup value is a pure convolution of the two
board distributions over DISJOINT dealt hands.  This supplies the cross term
of the M-C fixed point:

    E_ff(n1, n2)   immediate score, hero perspective
    stay probabilities, singly and jointly (hands are dependent through the
    shared deck, so the joint is measured rather than assumed independent)

The chain terms (who returns to FL at what count) are combined with these
numbers inside the fixed-point solver; this module only measures.

Usage:
    python -m ai.tutor.fl_vs_fl --library D:/ofc_data/fl_library_14 --pairs 200000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ai.tutor.t4_vs_fl import FlLibrary


def sample_disjoint_pairs(
    library: FlLibrary, want: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    size = len(library)
    left_out, right_out = [], []
    found = 0
    while found < want:
        batch = min(4_000_000, max(1_000_000, (want - found) * 160))
        left = rng.integers(0, size, batch)
        right = rng.integers(0, size, batch)
        ok = (library.masks[left] & library.masks[right]) == 0
        ok &= left != right
        left, right = left[ok], right[ok]
        left_out.append(left)
        right_out.append(right)
        found += len(left)
    return (
        np.concatenate(left_out)[:want],
        np.concatenate(right_out)[:want],
    )


def convolve(library: FlLibrary, pairs: int, seed: int) -> dict:
    left, right = sample_disjoint_pairs(library, pairs, seed)
    hero_values = library.values[left]
    opp_values = library.values[right]
    hero_royalty = library.royalty[left]
    opp_royalty = library.royalty[right]
    hero_stay = library.stay[left]
    opp_stay = library.stay[right]
    hero_busted = library.busted[left]
    opp_busted = library.busted[right]

    lines = np.sign(hero_values - opp_values).sum(axis=1)
    scoop = np.where(lines == 3, 3.0, np.where(lines == -3, -3.0, 0.0))
    alive = lines + scoop + hero_royalty - opp_royalty
    base = np.where(
        hero_busted & opp_busted,
        0.0,
        np.where(
            hero_busted,
            -6.0 - opp_royalty,
            np.where(opp_busted, 6.0 + hero_royalty, alive),
        ),
    )
    acceptance = None  # informational only; sampler is rejection-based
    return {
        "pairs": int(pairs),
        "immediate_score_mean": float(base.mean()),
        "immediate_score_std": float(base.std()),
        "score_se": float(base.std() / np.sqrt(pairs)),
        "stay_hero": float(hero_stay.mean()),
        "stay_opp": float(opp_stay.mean()),
        "stay_both": float((hero_stay & opp_stay).mean()),
        "stay_joint_minus_independent": float(
            (hero_stay & opp_stay).mean() - hero_stay.mean() * opp_stay.mean()
        ),
        "royalty_mean": float(hero_royalty.mean()),
        "acceptance_note": acceptance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    library = FlLibrary(args.library)
    report = convolve(library, args.pairs, args.seed)
    report["library"] = str(args.library)
    report["library_boards"] = len(library)
    report["schema"] = "ofc_fl_vs_fl_convolution/v1"
    report["counts"] = [14, 14]
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
