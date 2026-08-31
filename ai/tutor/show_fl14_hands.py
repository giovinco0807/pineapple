"""Play whole hands with the served chain and print them street by street.

Every other reader in this tree consumes a number.  This one exists so the
placements can be looked at: five cards on the opening, two a street after
that, and what the Fantasyland opponent does to the finished board.

The chain is the one it is handed -- T0/T1/T2 by model, T3 and T4 by the exact
solver, which is what plays those streets today.  Nothing here re-derives a
decision; it drives the same three binaries a measurement run drives and
reassembles their outputs into a board that changes as you read down.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ai.tutor.make_fl14_deals import card_names, deal

ROWS = ("top", "middle", "bottom")
CAP = (3, 5, 5)


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rows_of(text: str) -> list[list[str]]:
    return [[c for c in part.split(",") if c] for part in text.split("|")]


def board_lines(rows: list[list[str]], mark: set[str] | None = None) -> list[str]:
    out = []
    for index, name in enumerate(ROWS):
        cards = rows[index]
        shown = " ".join(f"[{c}]" if mark and c in mark else f" {c} " for c in cards)
        slots = "  .  " * (CAP[index] - len(cards))
        out.append(f"    {name:<7}{shown}{slots}")
    return out


def placed_between(before: list[list[str]], after: list[list[str]]) -> list[tuple[str, str]]:
    moves = []
    for index, name in enumerate(ROWS):
        remaining = list(before[index])
        for card in after[index]:
            if card in remaining:
                remaining.remove(card)
            else:
                moves.append((card, name))
    return moves


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=lambda v: int(v, 0), default=0xC0FFEE99)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--hands", type=int, default=5)
    parser.add_argument("--opponents", type=int, default=240)
    parser.add_argument("--pool", default="D:/ofc_data/fl_pools/fl14_v1.jfl1")
    parser.add_argument("--t0-model", default="D:/ofc_data/lap2_t0_model/evaluator.bin")
    parser.add_argument("--t1-model", default="D:/ofc_data/lap2_t1_model/evaluator.bin")
    parser.add_argument("--t2-model", default="D:/ofc_data/lap2_t2_model/evaluator.bin")
    parser.add_argument("--play", default=None)
    parser.add_argument("--solver", default=None)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    release = args.workspace_root / "ai" / "rust_solver" / "target" / "release"
    args.play = args.play or str(release / "t4_first_exact.exe")
    args.solver = args.solver or str(release / "fl_solver.exe")
    for binary in (args.play, args.solver):
        if not Path(binary).exists():
            raise SystemExit(f"no binary at {binary}")

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        # Seventeen cards: five to open and three a street for four streets.
        deals = {
            str(args.seed + index): card_names(deal(args.seed, index, 17))
            for index in range(args.offset, args.offset + args.hands)
        }
        play_in = work / "deals.jsonl"
        with play_in.open("w", encoding="utf-8", newline="\n") as handle:
            for ident, cards in deals.items():
                handle.write(json.dumps({"id": ident, "cards": cards[:14]}) + "\n")

        run([
            args.play, "--play-roots", "--input", str(play_in),
            "--output", str(work / "t3.jsonl"),
            "--play-t1-output", str(work / "t1.jsonl"),
            "--play-t2-output", str(work / "t2.jsonl"),
            "--play-t0-model", args.t0_model,
            "--play-t1-model", args.t1_model,
            "--play-t2-model", args.t2_model,
            "--chunk-size", str(max(args.hands, 1)),
        ])
        run([
            args.solver, "teach", "--pool", args.pool,
            "--roots-file", str(work / "t3.jsonl"),
            "--opponents", str(args.opponents),
            "--out-dir", str(work / "t3labels"),
        ])

        t1 = {r["id"]: r for r in read_jsonl(work / "t1.jsonl")}
        t2 = {r["id"]: r for r in read_jsonl(work / "t2.jsonl")}
        t3 = {r["id"]: r for r in read_jsonl(work / "t3.jsonl")}
        labels = {r["id"]: r for r in read_jsonl(work / "t3labels" / "t3_labels.jsonl")}

        # The T3 street is served by taking the solver's best action, which is
        # the eleven-card board plus the card it threw.
        t4_in = work / "t4.jsonl"
        chosen: dict[str, tuple[list[list[str]], str]] = {}
        with t4_in.open("w", encoding="utf-8", newline="\n") as handle:
            for ident, record in labels.items():
                best = max(record["actions"], key=lambda a: (a["value"], a["action_key"]))
                *rows_text, discard = best["action_key"].split("|")
                rows = [[c for c in part.split(",") if c] for part in rows_text]
                chosen[ident] = (rows, discard)
                dead = t3[ident]["dead"] + [discard]
                handle.write(
                    json.dumps(
                        {
                            "id": ident,
                            "rows": rows,
                            "dead": dead,
                            "draw": deals[ident][14:17],
                        }
                    )
                    + "\n"
                )
        run([
            args.solver, "finish-t4", "--pool", args.pool,
            "--roots-file", str(t4_in), "--opponents", str(args.opponents),
            "--out", str(work / "final.jsonl"),
        ])
        finals = {r["id"]: r for r in read_jsonl(work / "final.jsonl")}

    for ident, cards in deals.items():
        if ident not in finals:
            print(f"hand {ident}: no finished board (short draw)")
            continue
        final = finals[ident]
        print("=" * 62)
        print(f"hand {ident}   dealt {' '.join(cards)}")
        print("=" * 62)

        empty = [[], [], []]
        opening = [list(row) for row in t1[ident]["board"].values()]
        print(f"\n  T0  places five: {' '.join(cards[:5])}")
        for line in board_lines(opening, set(cards[:5])):
            print(line)

        before = opening
        for street, source, drawn in (
            ("T1", t2[ident], t1[ident]["draw"]),
            ("T2", t3[ident], t2[ident]["draw"]),
        ):
            after = [list(row) for row in source["rows"]]
            moves = placed_between(before, after)
            discard = [c for c in source["dead"] if c not in sum(before, [])]
            print(
                f"\n  {street}  draws {' '.join(drawn)}"
                f"   places {', '.join(f'{c} to {r}' for c, r in moves)}"
                f"   throws {discard[-1] if discard else '?'}"
            )
            for line in board_lines(after, {c for c, _ in moves}):
                print(line)
            before = after

        rows, discard = chosen[ident]
        moves = placed_between(before, rows)
        print(
            f"\n  T3  draws {' '.join(t3[ident]['draw'])}"
            f"   places {', '.join(f'{c} to {r}' for c, r in moves)}   throws {discard}"
            "   (exact solver)"
        )
        for line in board_lines(rows, {c for c, _ in moves}):
            print(line)

        finished = rows_of(final["best"].split("|")[0] + "|" + "|".join(final["best"].split("|")[1:3]))
        moves = placed_between(rows, finished)
        print(
            f"\n  T4  draws {' '.join(cards[14:17])}"
            f"   places {', '.join(f'{c} to {r}' for c, r in moves)}"
            "   (exact solver)"
        )
        for line in board_lines(finished, {c for c, _ in moves}):
            print(line)

        verdict = "FOUL" if final["busted"] else "stands"
        entry = final["entry_width"]
        print(
            f"\n  result: {verdict}   royalty {final['royalty']}"
            f"   Fantasyland {'no' if entry < 14 else str(entry) + ' cards'}"
            f"   score vs {args.opponents} FL opponents {final['value']:+.3f}"
        )
        print(
            f"  of this position's T4 placements, {final['unfouled']} do not foul"
        )
        print()


if __name__ == "__main__":
    main()
