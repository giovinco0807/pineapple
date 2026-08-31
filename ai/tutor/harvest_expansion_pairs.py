"""Harvest the pair distribution the T2 teacher actually reads.

The out-of-distribution defect (canonical doc, section 9): boundary nets are
trained on trace pairs -- both boards real play -- but read on expansion
pairs, where the hero board is any candidate placement and the opponent
board is any placement of a sampled draw.  This walks the T2-BB requests and
emits states from exactly that expansion distribution, so V3s can finally be
trained on what it will be asked about.

Two outputs per harvested pair:
  pairs.jsonl     -- the boundary state itself (board/dead/opp_board), for
                     encoding into V3s training rows
  probe.jsonl     -- the pair expanded into --draws-per-pair T3 decision
                     states for the Rust chooser teacher (t3-first-hu), whose
                     per-root values average into the pair's target.  The
                     teacher stands on exact V4, so the labels inherit no
                     boundary net -- the circularity the fix must avoid.

All sampling is hashlib-seeded: two runs harvest identical states.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from ai.tutor.encode_fl14_teacher import ALL_CARDS, CARD_INDEX, seen_mask
from ai.tutor.hu_street_teacher import apply_map, joker_maps, placements


def rng_for(*parts) -> random.Random:
    digest = hashlib.sha256("/".join(str(p) for p in parts).encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def fresh_jokers(cards: list[str], used: int) -> tuple[list[str], int]:
    """Rename raw sampled jokers past every name already in play."""
    out = []
    for card in cards:
        if card.startswith("X"):
            used += 1
            out.append(f"X{used}")
        else:
            out.append(card)
    return out, used


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--pairs-per-root", type=int, default=2)
    parser.add_argument("--draws-per-pair", type=int, default=4)
    parser.add_argument("--out-pairs", type=Path, required=True)
    parser.add_argument("--out-probe", type=Path, required=True)
    args = parser.parse_args()

    pairs_out = args.out_pairs.open("w", encoding="utf-8", newline="\n")
    probe_out = args.out_probe.open("w", encoding="utf-8", newline="\n")
    n_pairs = n_probe = 0
    for line in args.requests.open(encoding="utf-8"):
        if not line.strip():
            continue
        request = json.loads(line)
        hero_cards = ([c for row in request["board"] for c in row]
                      + request["dead"] + request["draw"])
        opp_cards = [c for row in request["opp_board"] for c in row]
        hero_map, opp_map = joker_maps(hero_cards, opp_cards)
        board = apply_map(request["board"], hero_map)
        draw = apply_map(request["draw"], hero_map)
        dead = apply_map(request["dead"], hero_map)
        opp_rows = apply_map(request["opp_board"], opp_map)
        jokers_used = len([c for c in hero_cards + opp_cards if c.startswith("X")])

        actions = placements(board, draw)
        all_seen = ([c for row in board for c in row]
                    + [c for row in opp_rows for c in row] + dead + draw)
        mask = seen_mask(all_seen)
        unseen = [c for c in ALL_CARDS if not ((1 << CARD_INDEX[c]) & mask)]

        rng = rng_for("harvest", request["id"])
        for k in range(args.pairs_per_root):
            hero_after, toss = actions[rng.randrange(len(actions))]
            opp_draw, used = fresh_jokers(rng.sample(unseen, 3), jokers_used)
            opp_options = placements(opp_rows, opp_draw)
            opp_after, _opp_toss = opp_options[rng.randrange(len(opp_options))]
            pair_id = f"{request['id']}p{k}"
            pair_dead = dead + [toss]
            pairs_out.write(json.dumps({
                "id": pair_id,
                "board": hero_after,
                "dead": pair_dead,
                "opp_board": opp_after,
            }, separators=(",", ":")) + "\n")
            n_pairs += 1
            # The pair's own unseen: sampled opp cards are now visible.
            pair_seen = mask
            for c in opp_draw:
                if not c.startswith("X"):
                    pair_seen |= 1 << CARD_INDEX[c]
            jokers_visible = used
            pair_unseen = [
                c for c in ALL_CARDS
                if not ((1 << CARD_INDEX[c]) & pair_seen)
                and not c.startswith("X")
            ]
            pair_unseen += [f"X{jokers_visible + 1 + j}"
                            for j in range(2 - min(2, jokers_visible))]
            for d in range(args.draws_per_pair):
                drng = rng_for("draw", pair_id, d)
                probe_out.write(json.dumps({
                    "id": f"{pair_id}d{d}",
                    "board": hero_after,
                    "dead": pair_dead,
                    "draw": drng.sample(pair_unseen, 3),
                    "opp_board": opp_after,
                    "opp_draws": 16,
                }, separators=(",", ":")) + "\n")
                n_probe += 1
    pairs_out.close()
    probe_out.close()
    print(f"{n_pairs} pairs, {n_probe} probe states")


if __name__ == "__main__":
    main()
