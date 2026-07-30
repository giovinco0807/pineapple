"""Generate T4-vs-FL teacher data from the global FL board library.

Labels are Monte Carlo means over the library boards compatible with each
root (uniform conditional by construction); the manifest records the mean
sample count so the label noise floor is explicit.  Roots below a minimum
sample count are skipped rather than labeled noisily.

Split is keyed on the root seed and fixed before labeling, as everywhere.

Usage:
    python -m ai.tutor.generate_t4_vs_fl_teacher --roots 40000 \
        --libraries D:/ofc_data/fl_library_14 --out-dir <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

import ai.tutor.exact_late as exact_late
from ai.engine.action_space import get_turn_actions
from ai.engine.encoding import ALL_CARDS, Board
from ai.tutor.t4_vs_fl import (
    CARD_INDEX,
    FEATURE_SIZE,
    FlLibrary,
    encode_action,
    hero_terminal,
    sample_root,
    score_against_library,
    seen_mask,
)

DATASET_SCHEMA = "ofc_t4_vs_fl_teacher/v1"
SPLITS = ("fit", "dev", "test")


def split_of(seed: int) -> str:
    digest = hashlib.sha256(f"t4-vs-fl-teacher-v1/{seed}".encode()).digest()
    bucket = int.from_bytes(digest[:4], "big") % 100
    return "fit" if bucket < 80 else ("dev" if bucket < 90 else "test")


class MergedLibrary(FlLibrary):
    def __init__(self, dirs: list[Path]) -> None:
        parts = [FlLibrary(d) for d in dirs if d.exists()]
        if not parts:
            raise FileNotFoundError(f"no libraries found in {dirs}")
        self.masks = np.concatenate([p.masks for p in parts])
        self.values = np.concatenate([p.values for p in parts])
        self.royalty = np.concatenate([p.royalty for p in parts])
        self.stay = np.concatenate([p.stay for p in parts])
        self.busted = np.concatenate([p.busted for p in parts])


def run(
    *,
    roots: int,
    seed: int,
    libraries: list[Path],
    out_dir: Path,
    opp_count: int,
    min_samples: int,
) -> dict:
    library = MergedLibrary(libraries)
    out_dir.mkdir(parents=True, exist_ok=True)
    buffers = {name: {"x": [], "y": [], "j": []} for name in SPLITS}
    sample_counts = []
    skipped = 0
    started = time.time()

    for index in range(roots):
        root = sample_root(seed + index, opp_count)
        cards_seen = (
            [card for row in root["board"] for card in row]
            + list(root["dead"])
            + list(root["draw"])
        )
        mask = seen_mask(cards_seen)
        indices = library.matching(mask)
        if len(indices) < min_samples:
            skipped += 1
            continue
        sample_counts.append(len(indices))
        pool_cards = [
            card for card in ALL_CARDS if not ((1 << CARD_INDEX[card]) & mask)
        ]
        board = Board(
            top=list(root["board"][0]),
            middle=list(root["board"][1]),
            bottom=list(root["board"][2]),
        )
        jokers_visible = sum(1 for card in cards_seen if card in ("X1", "X2"))
        target = buffers[split_of(root["seed"])]
        for action in get_turn_actions(list(root["draw"]), board):
            final = exact_late.apply_action(board, action)
            rows = (final.top, final.middle, final.bottom)
            terminal = hero_terminal(rows)
            value = score_against_library(terminal, library, indices, opp_count)
            target["x"].append(encode_action(rows, root, pool_cards))
            target["y"].append(value)
            target["j"].append(jokers_visible)
        if (index + 1) % 5000 == 0:
            rate = (index + 1) / (time.time() - started)
            print(f"[{index + 1}/{roots}] {rate:.0f} roots/s", flush=True)

    manifest = {
        "schema": DATASET_SCHEMA,
        "feature_size": FEATURE_SIZE,
        "label": "mc_mean_score_vs_fl_library",
        "opp_count": opp_count,
        "roots": roots,
        "skipped_low_sample_roots": skipped,
        "seed": seed,
        "libraries": [str(d) for d in libraries],
        "library_boards": int(len(library.masks)),
        "mean_fl_samples_per_root": float(np.mean(sample_counts)) if sample_counts else 0,
        "min_fl_samples": int(np.min(sample_counts)) if sample_counts else 0,
        "label_se_estimate": float(6.0 / np.sqrt(np.mean(sample_counts)))
        if sample_counts
        else None,
        "split_rule": "sha256('t4-vs-fl-teacher-v1/<seed>') % 100 -> 80/10/10",
        "elapsed_seconds": time.time() - started,
        "splits": {},
    }
    for name in SPLITS:
        x = np.asarray(buffers[name]["x"], dtype=np.float32)
        y = np.asarray(buffers[name]["y"], dtype=np.float32)
        j = np.asarray(buffers[name]["j"], dtype=np.int8)
        np.savez_compressed(out_dir / f"{name}.npz", x=x, y=y, jokers=j)
        manifest["splits"][name] = {
            "rows": int(x.shape[0]),
            "ev_mean": float(y.mean()) if y.size else None,
            "ev_std": float(y.std()) if y.size else None,
        }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=int, default=40_000)
    parser.add_argument("--seed", type=int, default=45_000_000)
    parser.add_argument("--libraries", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--opp-count", type=int, default=14)
    parser.add_argument("--min-samples", type=int, default=25)
    args = parser.parse_args()
    manifest = run(
        roots=args.roots,
        seed=args.seed,
        libraries=args.libraries,
        out_dir=args.out_dir,
        opp_count=args.opp_count,
        min_samples=args.min_samples,
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "splits"}, indent=2))
    print(json.dumps(manifest["splits"], indent=2))


if __name__ == "__main__":
    main()
