"""Generic HU street teacher: label a street's decisions by reading the
street-start value net one level down.

The ladder's shape, per street k and seat:

* the second seat's (BTN's) placement ends the street, so its after-state IS
  the next street's start and each action is **one value read**:
      label = -V(opp board, hero after-board)        (zero-sum flip)
* the first seat's (BB's) placement leaves the opponent's half-street, so
  each action expands the opponent's sampled draws, lets a chooser place
  them, and reads the value at the boundary:
      label = E[opp draw]( chooser -> V(hero after-board, opp after-board) )

Everything heavy is batched: candidate boards are encoded through the same
block binary the evaluators train on (one subprocess per batch, parallel
inside), and the value net runs one torch call per batch.  The chooser for
the opponent's half-street is the *value net itself* ranking the opponent's
candidates from the opponent's perspective -- the measured chooser tolerance
(errors ride common-mode under shared draws) is what licenses that.

The value net V is a street-start net: input = the 207-dim pair encoding of
(first-actor board, second-actor board) at the street boundary, output = the
first actor's value.  For the T3 boundary it is trained straight from the
T3-first teacher's root maxima; each street trained here then supplies the
boundary net for the street above.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np

from ai.tutor.encode_fl14_teacher import (
    ALL_CARDS,
    CARD_INDEX,
    actor_block,
    allocation_rank_block,
    context_block,
    fetch_blocks,
    seen_mask,
)
from ai.tutor.encode_hu_teacher import canonical_jokers
from ai.tutor.solver_paths import _solver_path

ROW_CAPACITY = [3, 5, 5]


def joker_maps(hero_cards: list[str], opp_cards: list[str]) -> tuple[dict, dict]:
    """Seat-local joker names to globally distinct ones.

    Each seat names its own first joker X1, so the two seats' X1 are two
    different physical cards and a shared pool must tell them apart.  The maps
    are built from the *sets* of names a seat uses, not by walking cards, so a
    board and its own later superset (opp_board inside opp_after) rename
    identically instead of counting the same joker twice.
    """
    hero_names = sorted({c for c in hero_cards if c.startswith("X")})
    opp_names = sorted({c for c in opp_cards if c.startswith("X")})
    if len(hero_names) + len(opp_names) > 2:
        raise ValueError(
            f"{len(hero_names)} + {len(opp_names)} jokers; the deck holds two"
        )
    hero_map = {name: f"X{i + 1}" for i, name in enumerate(hero_names)}
    opp_map = {
        name: f"X{len(hero_names) + i + 1}" for i, name in enumerate(opp_names)
    }
    return hero_map, opp_map


def apply_map(rows, mapping: dict):
    if rows and isinstance(rows[0], list):
        return [[mapping.get(c, c) for c in row] for row in rows]
    return [mapping.get(c, c) for c in rows]


def rows_key(rows: list[list[str]], discard: str) -> str:
    parts = [",".join(sorted(row)) for row in rows]
    parts.append(discard)
    return "|".join(parts)


def t0_arrangements(draw: list[str]) -> list[tuple[list[list[str]], str]]:
    """Every distinct arrangement of five dealt cards; no discard at T0."""
    from itertools import combinations
    out = []
    seen: set[str] = set()
    cards = list(draw)
    indices = set(range(5))
    for top_n in range(0, 4):
        for top in combinations(sorted(indices), top_n):
            rest1 = indices - set(top)
            for mid_n in range(0, min(5, len(rest1)) + 1):
                for mid in combinations(sorted(rest1), mid_n):
                    bot = rest1 - set(mid)
                    if len(bot) > 5:
                        continue
                    rows = [
                        [cards[i] for i in top],
                        [cards[i] for i in mid],
                        [cards[i] for i in sorted(bot)],
                    ]
                    key = rows_key(rows, "")
                    if key not in seen:
                        seen.add(key)
                        out.append((rows, ""))
    return out


def placements(board: list[list[str]], draw: list[str]) -> list[tuple[list[list[str]], str]]:
    out = []
    seen: set[str] = set()
    for toss in range(3):
        kept = [k for k in range(3) if k != toss]
        for row_a in range(3):
            for row_b in range(3):
                need = [0, 0, 0]
                need[row_a] += 1
                need[row_b] += 1
                if any(len(board[r]) + need[r] > ROW_CAPACITY[r] for r in range(3)):
                    continue
                after = [list(r) for r in board]
                after[row_a].append(draw[kept[0]])
                after[row_b].append(draw[kept[1]])
                key = rows_key(after, draw[toss])
                if key not in seen:
                    seen.add(key)
                    out.append((after, draw[toss]))
    return out


class PairEncoder:
    """The 207-dim (first-actor, second-actor) boundary encoding, batched.

    Block requests are deduplicated on (board, pool).  Within a root the
    opponent's board never changes and hero's discard takes only three
    values, so the opponent's blocks are three computations rather than one
    per action -- the dominant cost before this, since a joint block over
    four open slots is four hundred sampled completions.
    """

    def __init__(self, workspace_root: Path, joint_samples: int = 400):
        self.solver = str(_solver_path(workspace_root))
        self.workspace_root = workspace_root
        self.joint_samples = joint_samples

    def encode(self, pairs: list[tuple[list[list[str]], list[list[str]], list[str]]]) -> np.ndarray:
        requests = []
        pools = []
        seen_keys: dict[tuple, str] = {}
        slot_of: list[tuple[str, str]] = []
        for position, (first, second, extra_seen) in enumerate(pairs):
            all_seen = (
                [c for row in first for c in row]
                + [c for row in second for c in row]
                + extra_seen
            )
            if bin(seen_mask(all_seen)).count("1") != len(all_seen):
                raise AssertionError(f"a card repeats in {all_seen}")
            seen = seen_mask(all_seen)
            pool = [c for c in ALL_CARDS if not ((1 << CARD_INDEX[c]) & seen)]
            pools.append(pool)
            ids = []
            for rows in (first, second):
                key = (
                    tuple(tuple(sorted(row)) for row in rows),
                    tuple(pool),
                )
                if key not in seen_keys:
                    block_id = str(len(seen_keys))
                    seen_keys[key] = block_id
                    requests.append(
                        {
                            "id": block_id,
                            # The seed rides the board, so identical requests
                            # stay identical: a sampled joint block must not
                            # depend on which action asked for it.
                            "seed": block_id,
                            "board": {
                                "top": rows[0],
                                "middle": rows[1],
                                "bottom": rows[2],
                            },
                            "pool": pool,
                        }
                    )
                ids.append(seen_keys[key])
            slot_of.append((ids[0], ids[1]))
        blocks = fetch_blocks(
            requests, self.workspace_root, self.joint_samples, self.solver
        )
        vectors = []
        for position, (first, second, _extra) in enumerate(pairs):
            a_id, b_id = slot_of[position]
            a_rowwise, a_joint = blocks[a_id]
            b_rowwise, b_joint = blocks[b_id]
            pool = pools[position]
            a_actor, _ = actor_block(first, pool)
            b_actor, _ = actor_block(second, pool)
            vectors.append(
                a_actor
                + [float(v) for v in a_rowwise]
                + [float(v) for v in a_joint]
                + context_block(pool)
                + allocation_rank_block(first)
                + b_actor
                + [float(v) for v in b_rowwise]
                + [float(v) for v in b_joint]
            )
        return np.asarray(vectors, dtype=np.float32)


def prepare_request(request: dict, seat: str, opp_draws_wanted: int,
                    stream_offset: int) -> dict:
    """Everything a request's branches need, computed exactly once.

    The m1->m3 collapse traced to two branches that were supposed to mirror
    each other and did not (canonical doc, section 10).  Both passes of the
    teacher -- the chooser pre-pass and the pair building -- now consume this
    one context instead of re-deriving any of it.
    """
    hero_cards = ([c for row in request["board"] for c in row]
                  + request["dead"] + request["draw"])
    opp_cards = [c for row in request["opp_board"] for c in row]
    if "opp_after" in request:
        opp_cards = opp_cards + [c for row in request["opp_after"] for c in row]
    hero_map, opp_map = joker_maps(hero_cards, opp_cards)
    own_board = apply_map(request["board"], hero_map)
    draw = apply_map(request["draw"], hero_map)
    dead = apply_map(request["dead"], hero_map)
    opp_rows = apply_map(request["opp_board"], opp_map)
    actions = (
        t0_arrangements(draw) if len(draw) == 5 else placements(own_board, draw)
    )
    context = {
        "own_board": own_board, "draw": draw, "dead": dead,
        "opp_rows": opp_rows, "actions": actions,
        "opp_after_rows": (
            apply_map(request["opp_after"], opp_map)
            if "opp_after" in request else None
        ),
        "opp_draws": [], "opp_options": [],
    }
    if seat == "bb":
        all_seen = ([c for row in own_board for c in row]
                    + [c for row in opp_rows for c in row] + dead + draw)
        seen = seen_mask(all_seen)
        unseen = [c for c in ALL_CARDS if not ((1 << CARD_INDEX[c]) & seen)]
        seed_bytes = hashlib.sha256(
            f"{request['id']}/{stream_offset}".encode()
        ).digest()
        rng = random.Random(int.from_bytes(seed_bytes[:8], "big"))
        context["opp_draws"] = [
            rng.sample(unseen, 3) for _ in range(opp_draws_wanted)
        ]
        context["opp_options"] = [
            placements(opp_rows, opp_draw) for opp_draw in context["opp_draws"]
        ]
        context["unseen"] = unseen
    return context


class OppChooser:
    """The opponent placing its own draw the way it actually plays.

    Scores each placement with the own-hand ranker (96 dims: actor +
    rowwise + context, the joint-free lap4 search encoding) and keeps the
    argmax.  This is the Rust T3 teacher's opponent model, ported: it matches
    the real gen-3 opponent instead of a best-responding adversary, and it
    never reads the value net on the adversarial tail where that net is
    measured +5 too optimistic.
    """

    def __init__(self, model_path: Path, workspace_root: Path):
        self.net = ValueNet(model_path)
        self.solver = str(_solver_path(workspace_root))
        self.workspace_root = workspace_root

    def choose(self, contexts: list[dict]) -> dict:
        """chosen[(context_index, draw_index)] = the opponent's after-board."""
        requests, meta = [], []
        for c_index, context in enumerate(contexts):
            for d_index, options in enumerate(context["opp_options"]):
                pool = [c for c in context["unseen"]
                        if c not in context["opp_draws"][d_index]]
                for j, (opp_after, _toss) in enumerate(options):
                    block_id = str(len(requests))
                    requests.append({
                        "id": block_id, "seed": block_id,
                        "board": {"top": opp_after[0], "middle": opp_after[1],
                                  "bottom": opp_after[2]},
                        "pool": pool,
                    })
                    meta.append((c_index, d_index, j, opp_after, pool))
        if not requests:
            return {}
        blocks = fetch_blocks(requests, self.workspace_root, 1, self.solver)
        vectors = []
        for block_id, (_c, _d, _j, opp_after, pool) in enumerate(meta):
            rowwise, _joint = blocks[str(block_id)]
            actor, _ = actor_block(opp_after, pool)
            vectors.append(
                actor + [float(v) for v in rowwise] + context_block(pool)
            )
        scores = self.net.predict(np.asarray(vectors, dtype=np.float32))
        best: dict = {}
        for (c_index, d_index, _j, opp_after, _pool), score in zip(meta, scores):
            key = (c_index, d_index)
            if key not in best or score > best[key][0]:
                best[key] = (score, opp_after)
        return {key: board for key, (_s, board) in best.items()}


class ValueNet:
    """The boundary net, from a torch checkpoint or an exported `.npz`.

    The network is a plain ReLU stack, so its forward pass is six lines of
    numpy.  That matters off this workstation: a fleet worker that must
    install torch pays 800 MB and two minutes of boot for one matrix
    multiply per batch, and the existing label workers deliberately install
    nothing but `python3-numpy`.  `export_value_npz.py` writes the `.npz`;
    the two paths are checked against each other there, not asserted here.
    """

    def __init__(self, checkpoint: Path):
        if str(checkpoint).endswith(".npz"):
            payload = np.load(checkpoint)
            self.mean = payload["input_mean"].astype(np.float32)
            self.std = payload["input_std"].astype(np.float32)
            self.layers = []
            index = 0
            while f"w{index}" in payload:
                self.layers.append(
                    (payload[f"w{index}"].astype(np.float32),
                     payload[f"b{index}"].astype(np.float32))
                )
                index += 1
            if not self.layers:
                raise ValueError(f"{checkpoint} holds no layers")
            return
        import torch

        from ai.tutor.train_t4_first_evaluator import T4FirstEvaluator

        c = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = T4FirstEvaluator(c["input_dim"], tuple(c["hidden"]))
        model.load_state_dict(c["model_state_dict"])
        model.eval()
        self.mean = np.asarray(c["input_mean"], dtype=np.float32)
        self.std = np.asarray(c["input_std"], dtype=np.float32)
        self.layers = [
            (layer.weight.detach().numpy().astype(np.float32),
             layer.bias.detach().numpy().astype(np.float32))
            for layer in model.net
            if hasattr(layer, "weight")
        ]

    def predict(self, x: np.ndarray) -> np.ndarray:
        h = (x - self.mean) / self.std
        for index, (weight, bias) in enumerate(self.layers):
            h = h @ weight.T + bias
            if index + 1 < len(self.layers):
                np.maximum(h, 0.0, out=h)
        return h.reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True,
                        help="street rows: {id, board, dead, draw, opp_board}")
    parser.add_argument("--seat", choices=["btn", "bb"], required=True)
    parser.add_argument("--value-model", type=Path, required=True,
                        help="boundary net for the street below")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--opp-draws", type=int, default=8,
                        help="bb only: opponent draws sampled per action")
    parser.add_argument("--stream-offset", type=int, default=0)
    parser.add_argument("--batch-roots", type=int, default=64)
    parser.add_argument("--joint-samples", type=int, default=400,
                        help="completions sampled per joint block")
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--start", type=int, default=0,
                        help="first request line to label")
    parser.add_argument("--count", type=int, default=0,
                        help="requests to label from --start; 0 means all")
    parser.add_argument("--traced-opponent", dest="traced", action="store_true",
                        default=True)
    parser.add_argument("--no-traced-opponent", dest="traced",
                        action="store_false",
                        help="bb only: expand --opp-draws sampled opponent "
                             "half-streets instead of reading the one the "
                             "trace played")
    parser.add_argument("--opp-chooser", type=Path, default=None,
                        help="own-hand ranker (npz) for --opp-agg chooser")
    parser.add_argument("--opp-agg", choices=["min", "mean", "chooser"],
                        default="min",
                        help="bb expansion only: how the opponent's placements "
                             "collapse.  `min` is the best-responding "
                             "adversary through the value net's eyes -- which "
                             "both assumes an opponent that does not exist "
                             "(gen-3 never looks across the table) and "
                             "harvests the net's downward noise.  `mean` is "
                             "an indifferent opponent; the m3 post-mortem's "
                             "discriminating control.")
    args = parser.parse_args()

    encoder = PairEncoder(args.workspace_root, args.joint_samples)
    net = ValueNet(args.value_model)
    chooser = None
    if args.opp_agg == "chooser":
        if args.opp_chooser is None:
            raise SystemExit("--opp-agg chooser requires --opp-chooser")
        chooser = OppChooser(args.opp_chooser, args.workspace_root)

    requests = [json.loads(l) for l in args.requests.open(encoding="utf-8") if l.strip()]
    if args.count:
        requests = requests[args.start:args.start + args.count]
    elif args.start:
        requests = requests[args.start:]
    done = 0
    import time
    started = time.time()
    with args.out.open("w", encoding="utf-8", newline="\n") as out:
        for start in range(0, len(requests), args.batch_roots):
            batch = requests[start:start + args.batch_roots]
            contexts = [
                prepare_request(request, args.seat, args.opp_draws,
                                args.stream_offset)
                for request in batch
            ]
            chosen = {}
            if chooser is not None:
                expanding = [
                    (r_index, context)
                    for r_index, (request, context) in enumerate(zip(batch, contexts))
                    if args.seat == "bb"
                    and not (args.traced and context["opp_after_rows"] is not None)
                ]
                picked = chooser.choose([c for _r, c in expanding])
                for slot, (r_index, _c) in enumerate(expanding):
                    for (c_index, d_index), board in picked.items():
                        if c_index == slot:
                            chosen[(r_index, d_index)] = board
            # Build every pair this batch needs, then one encode + one net call.
            pair_specs = []   # (root_index, action_index, opp_draw_index)
            pairs = []
            root_actions = []
            for r_index, request in enumerate(batch):
                context = contexts[r_index]
                own_board = context["own_board"]
                draw = context["draw"]
                dead = context["dead"]
                opp_rows = context["opp_rows"]
                actions = context["actions"]
                root_actions.append((request, actions, dead))
                opp_after_rows = context["opp_after_rows"]
                if args.seat == "bb" and opp_after_rows is not None and args.traced:
                    # Lap-one shortcut: the opponent's half-street is the one
                    # placement it actually made in the trace -- an unbiased
                    # single sample of the expansion, absorbed by volume.
                    for a_index, (after, toss) in enumerate(actions):
                        extra = dead + ([toss] if toss else [])
                        pairs.append((after, opp_after_rows, extra))
                        pair_specs.append((r_index, a_index, 0))
                    continue
                if args.seat == "btn":
                    for a_index, (after, toss) in enumerate(actions):
                        # Boundary pair: (first actor = opponent, second = hero).
                        extra = dead + ([toss] if toss else [])
                        pairs.append((opp_rows, after, extra))
                        pair_specs.append((r_index, a_index, 0))
                else:
                    # Opponent's half-street, from the prepared context (the
                    # hashlib-seeded draws live in prepare_request).
                    for a_index, (after, toss) in enumerate(actions):
                        for d_index in range(len(context["opp_draws"])):
                            if chooser is not None:
                                # The opponent placed its draw its own way;
                                # one pair per draw.
                                pairs.append((after, chosen[(r_index, d_index)],
                                              dead + [toss]))
                                pair_specs.append((r_index, a_index, d_index))
                                continue
                            for opp_after, _t in context["opp_options"][d_index]:
                                # min/mean collapse: the opponent's choice among
                                # its candidates through the value net's eyes.
                                pairs.append((after, opp_after, dead + [toss]))
                                pair_specs.append((r_index, a_index, d_index))
            x = encoder.encode(pairs)
            values = net.predict(x)
            # Group the pair rows once.  Scanning `pair_specs` per action is
            # quadratic in the batch, which is invisible at T1 (27 actions a
            # root) and 496 million comparisons a batch at T0 (232).
            by_action: dict[tuple[int, int], dict[int, list[int]]] = {}
            for k, (ri, ai, di) in enumerate(pair_specs):
                by_action.setdefault((ri, ai), {}).setdefault(di, []).append(k)
            for r_index, (request, actions, _dead) in enumerate(root_actions):
                out_actions = []
                # `direct` must mirror the PAIR-BUILDING branch exactly.
                # It once tested only for opp_after's presence -- true for
                # every trace-derived request -- so --no-traced-opponent
                # built the full expansion and then read pairs[0]: one
                # arbitrary placement of one sampled draw, silently, for
                # every "16-draw" label ever produced.  That single line is
                # the m1->m3 match collapse (canonical doc section 10).
                direct = args.seat == "btn" or (
                    args.traced and "opp_after" in request
                )
                if direct:
                    sign = -1.0 if args.seat == "btn" else 1.0
                    for a_index, (after, toss) in enumerate(actions):
                        mask = by_action[(r_index, a_index)][0]
                        value = sign * float(values[mask[0]])
                        out_actions.append({"action_key": rows_key(after, toss), "value": value})
                else:
                    for a_index, (after, toss) in enumerate(actions):
                        totals = []
                        draws_seen = by_action[(r_index, a_index)]
                        for di, ks in sorted(draws_seen.items()):
                            draw_values = [float(values[k]) for k in ks]
                            if args.opp_agg == "min":
                                # Best-responding adversary (see --opp-agg).
                                totals.append(min(draw_values))
                            else:
                                totals.append(sum(draw_values) / len(draw_values))
                        out_actions.append({
                            "action_key": rows_key(after, toss),
                            "value": sum(totals) / len(totals),
                        })
                out_actions.sort(key=lambda a: a["action_key"])
                out.write(json.dumps({
                    "id": request["id"],
                    "schema": f"ofc_hu_street_teacher_{args.seat}/v1",
                    "actions": out_actions,
                }, separators=(",", ":")) + "\n")
            done += len(batch)
            rate = (time.time() - started) / done
            print(f"[{done}/{len(requests)}] {rate*1000:.0f} ms/root", flush=True)


if __name__ == "__main__":
    main()
