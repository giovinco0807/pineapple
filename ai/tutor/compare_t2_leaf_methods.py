"""Compare the ways a T2 label can treat the T3 street, on the same roots.

Three arms answer one question -- what does replacing the enumerated T3 street
with a learned model cost:

    A   `fl_solver teach-t2`: every T3 placement priced, the max taken.
    B1  the T3 model's own best score stands in for the line.
    B2  the T3 model only picks the placement; that line is played out and
        priced against the drawn pool entries.

A second pass of A on an independent opponent stream supplies the floor.  No
model can be asked to beat the teacher's disagreement with itself, and without
that number a difference between arms cannot be read.

Charging is asymmetric on purpose: an arm's *choice* is charged against the
reference's *values*, because a teacher is consumed by which action it calls
best.  Each arm is charged against both reference passes and the two are
reported, since either pass alone is one sample of the opponents.

The two labelers write different action keys -- `teach-t2` writes the rows the
action produces, the playout writes the placements it made -- so both are
reduced to `(discard, sorted (card, row) placements)` against the root board.
Jokers normalise to `X`: the labelers already deduplicate them by identity, so
the collapse cannot merge two distinct actions.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

ROWS = ("top", "middle", "bottom")


def norm_card(card: str) -> str:
    return "X" if card.startswith("X") else card


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def canonical(discard: str, placements: list[tuple[str, str]]) -> str:
    body = ",".join(f"{card}@{row}" for card, row in sorted(placements))
    return f"{norm_card(discard)}|{body}"


def teach_t2_key(root_rows: list[list[str]], action_key: str) -> str:
    """`top|middle|bottom|discard`, differenced against the root board."""
    fields = action_key.split("|")
    if len(fields) != 4:
        raise ValueError(f"unexpected teach-t2 key: {action_key}")
    placements: list[tuple[str, str]] = []
    for index, row_name in enumerate(ROWS):
        after = [norm_card(c) for c in fields[index].split(",") if c]
        before = [norm_card(c) for c in root_rows[index]]
        remaining = list(before)
        for card in after:
            if card in remaining:
                remaining.remove(card)
            else:
                placements.append((card, row_name))
        if remaining:
            raise ValueError(f"action drops a placed card: {action_key}")
    if len(placements) != 2:
        raise ValueError(f"expected two placements, got {placements}")
    return canonical(fields[3], placements)


def playout_key(action_key: str) -> str:
    payload = json.loads(action_key)
    placements = [(norm_card(card), row) for card, row in payload["placements"]]
    return canonical(payload["discard"], placements)


def load_arm(
    path: Path, roots: dict[str, list[list[str]]], style: str, tolerance: float = 1e-9
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Canonical values per root, plus what the joker collapse merged.

    `teach-t2` deduplicates the two jokers and the playout's `legal_actions`
    does not, so a draw holding both emits each placement twice -- once per
    labelling of interchangeable cards.  Merging is only sound if those twins
    carry the same value, so the spread is measured rather than assumed.
    """
    out: dict[str, dict[str, float]] = {}
    merged = 0
    worst_spread = 0.0
    for record in read_jsonl(path):
        ident = str(record["id"])
        collected: dict[str, list[float]] = defaultdict(list)
        for action in record["actions"]:
            if style == "teach":
                key = teach_t2_key(roots[ident], action["action_key"])
            else:
                key = playout_key(action["action_key"])
            collected[key].append(float(action["value"]))
        values: dict[str, float] = {}
        for key, seen in collected.items():
            spread = max(seen) - min(seen)
            worst_spread = max(worst_spread, spread)
            if spread > tolerance:
                raise ValueError(
                    f"{ident}: joker-twin actions disagree by {spread:.6g} in {path.name}"
                )
            merged += len(seen) - 1
            values[key] = sum(seen) / len(seen)
        out[ident] = values
    return out, {"joker_twins_merged": merged, "worst_twin_spread": worst_spread}


def charge(choice: dict[str, dict[str, float]], truth: dict[str, dict[str, float]]) -> dict[str, float]:
    """What `choice`'s pick costs, judged by `truth`'s values."""
    regrets: list[float] = []
    top1 = 0
    for ident, picks in choice.items():
        reference = truth.get(ident)
        if reference is None:
            continue
        shared = sorted(set(picks) & set(reference))
        if len(shared) < 2:
            continue
        pick = max(shared, key=lambda key: (picks[key], key))
        best = max(reference[key] for key in shared)
        regrets.append(best - reference[pick])
        top1 += reference[pick] == best
    regrets.sort()
    count = max(len(regrets), 1)
    return {
        "roots": len(regrets),
        "regret": sum(regrets) / count,
        "top1": top1 / count,
        "p90": regrets[int(0.90 * (len(regrets) - 1))] if regrets else 0.0,
        "p99": regrets[int(0.99 * (len(regrets) - 1))] if regrets else 0.0,
        "max": regrets[-1] if regrets else 0.0,
    }


def value_agreement(arm: dict[str, dict[str, float]], truth: dict[str, dict[str, float]]) -> dict[str, float]:
    """Root-centred value error: the level offset cannot change a ranking."""
    errors: list[float] = []
    correlations: list[float] = []
    # Uncentred, because the level is the interesting part here: arm A takes a
    # max over twenty-one placements each priced on twenty-four sampled T4
    # draws, and a max over noisy estimates is biased upward.  An arm that
    # prices one chosen placement does not pay that bias, so a negative signed
    # difference is expected and is not by itself an error.
    signed: list[float] = []
    for ident, values in arm.items():
        reference = truth.get(ident)
        if reference is None:
            continue
        shared = sorted(set(values) & set(reference))
        if len(shared) < 2:
            continue
        a = [values[key] for key in shared]
        b = [reference[key] for key in shared]
        signed.extend(x - y for x, y in zip(a, b))
        mean_a, mean_b = sum(a) / len(a), sum(b) / len(b)
        centred_a = [value - mean_a for value in a]
        centred_b = [value - mean_b for value in b]
        errors.extend(abs(x - y) for x, y in zip(centred_a, centred_b))
        norm_a = math.sqrt(sum(x * x for x in centred_a))
        norm_b = math.sqrt(sum(y * y for y in centred_b))
        if norm_a > 0 and norm_b > 0:
            correlations.append(sum(x * y for x, y in zip(centred_a, centred_b)) / (norm_a * norm_b))
    return {
        "actions": len(errors),
        "centred_mae": sum(errors) / max(len(errors), 1),
        "mean_signed_level_difference": sum(signed) / max(len(signed), 1),
        "within_root_correlation": sum(correlations) / max(len(correlations), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=Path, required=True)
    parser.add_argument("--reference-a", type=Path, required=True, help="teach-t2, stream one")
    parser.add_argument("--reference-b", type=Path, required=True, help="teach-t2, second stream")
    parser.add_argument("--arm", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    roots = {str(row["id"]): row["rows"] for row in read_jsonl(args.roots)}
    reference_a, merge_a = load_arm(args.reference_a, roots, "teach")
    reference_b, _ = load_arm(args.reference_b, roots, "teach")

    report: dict[str, Any] = {
        "schema": "ofc_fl14_t2_leaf_comparison/v1",
        "roots": len(roots),
        "reference_merge": merge_a,
        "floor": {
            "a_choice_on_b": charge(reference_a, reference_b),
            "b_choice_on_a": charge(reference_b, reference_a),
            "value": value_agreement(reference_a, reference_b),
        },
        "arms": {},
    }
    floor = (
        report["floor"]["a_choice_on_b"]["regret"] + report["floor"]["b_choice_on_a"]["regret"]
    ) / 2
    report["floor"]["symmetrised_regret"] = floor

    for entry in args.arm:
        name, _, path = entry.partition("=")
        arm, merge = load_arm(Path(path), roots, "playout")
        on_a = charge(arm, reference_a)
        on_b = charge(arm, reference_b)
        mean_regret = (on_a["regret"] + on_b["regret"]) / 2
        report["arms"][name] = {
            "charged_on_reference_a": on_a,
            "charged_on_reference_b": on_b,
            "mean_regret": mean_regret,
            "gap_over_floor": mean_regret - floor,
            "value_vs_reference_a": value_agreement(arm, reference_a),
            "merge": merge,
        }

    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
