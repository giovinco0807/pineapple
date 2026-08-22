"""Relabel a T3 corpus scoring the hero against ONE opponent instead of two.

    python scripts/relabel_one_opponent.py --corpus in.jsonl --output out.jsonl --workers 14

Everything else is held fixed: the same roots, the same deals, the same three
players acting, the same sample count, the same seed.  Only the hero's SCORE
drops its second pair term.  That isolates one question -- is the residual
regret caused by the label being a sum over two opponents, which forces the
argmax to compromise between two objectives that can point different ways?
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from measure_three_max_t3_floor import observation_of  # noqa: E402
from ofc_regular.action_key import action_key  # noqa: E402
from ofc_regular.three_max.exact import evaluate_t3  # noqa: E402


def _relabel(args: tuple) -> str:
    line, samples, fl_ev_14 = args
    record = json.loads(line)
    ranked = evaluate_t3(
        observation_of(record),
        samples=samples,
        seed=record["seed"],
        fl_ev_per_pair={14: fl_ev_14},
        score_opponents=1,
    )
    record["actions"] = [
        {
            "placements": [[c, r] for c, r in c_.action.placements],
            "discards": list(c_.action.discards),
            "key": action_key(c_.action).to_token(),
            "ev": c_.ev,
            "board": {
                "top": list(c_.board.top),
                "middle": list(c_.board.middle),
                "bottom": list(c_.board.bottom),
            },
        }
        for c_ in ranked
    ]
    record["score_opponents"] = 1
    record["score_gap"] = (
        ranked[0].ev - ranked[1].ev if len(ranked) > 1 else 0.0
    )
    record["ev_holdout"] = None
    return json.dumps(record, sort_keys=True, separators=(",", ":"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--fl-ev-14", type=float, default=9.6)
    args = parser.parse_args()

    lines = args.corpus.read_text(encoding="utf-8").splitlines()
    print(f"relabelling {len(lines)} roots against one opponent", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(
            pool.map(
                _relabel,
                [(line, args.samples, args.fl_ev_14) for line in lines],
                chunksize=8,
            )
        )
    args.output.write_text("\n".join(results) + "\n", encoding="utf-8")
    print(f"wrote {len(results)} rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
