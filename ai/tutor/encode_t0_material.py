"""Encode every opening of every audited T0 root into the 207-dim serving vector.

Seat-aware sibling of `encode_fl_material.py` for the normal-track T0
evaluators (`hu/t0_bb.bin`, `hu/t0_btn.bin`).  The material comes from
`t0_mine.py` (BB or `--seat btn`): per root, the served pick, the referee's
nominee, the race means over the audited field (`sel_means`) and the verdict.
This writes one 207-dim vector per (root, opening) so the correction trainer
works on arrays instead of re-invoking the solver.

# Why the full opening set, not just the audited candidates

`sel_means` holds the field the race actually ran (fence + strata + served).
Training only needs those.  But "did the correction move the SERVED pick" is
a question about the argmax over every opening, including the ones nomination
never scored -- a net that reorders the audited field while promoting an
unaudited placement has not been fixed, it has been broken somewhere nobody
looked.  So every opening is encoded and the audited ones are flagged; rows
outside the field carry `mean: null`.

# Seat

BB decides from an empty table: the opponent half of the pair vector is the
unplayed board (its joint block is the eight-zero rule, its rowwise block is
the empty board's outlook) and the served fence is `policy.bin`'s top-8.

BTN decides after BB has placed five: the opponent half is REALLY computed --
actor/rowwise over the shared pool plus a sampled joint block -- and there is
no policy fence (policy.bin is a BB-only net), so the served fence is the
ranker's top-K.  The decision state is the pair (hero's five, BB's board) and
the pool is everything but those ten cards.

# Parity, and which joint spec is encoded

The candidate list and its serving scores come from the binary's own
`--hu-t0-deep --rollouts 0` (`hu_match::t0_model_scores`), the same command
the audit ran.  That path encodes hero's board and the opponent tail with the
ARM's serve-time joint sample count (200, `--serve-joint-samples 200` in the
audit) under the node seed `model-rank/t0`.  `--hu-encode` cannot reproduce
that column: it pins the training-spec 400 samples and seeds by request id.
`--joint-outlook` calls the same `sampled_joint_block` with caller-chosen
samples and seed, so that is the path used here, at the SERVE spec by
default (`--joint-samples 200 --joint-seed model-rank/t0`).  The T0-BB design
note (docs/t0_evaluator_teaching_design_20260830.md) encoded at the training
spec 400x32 and had no parity check; this module prefers the served vector,
because the argmax being corrected is the one serving computes and float32
agreement with the served score column is the only proof that the Python
assembly and `hu_encode::board_blocks` build the same vector.  Pass
`--joint-samples 400` to reproduce the BB choice; the parity gate is then
relaxed to a report (the joint dims differ by sampling, as they must).

Usage:
    python -m ai.tutor.encode_t0_material --seat btn \\
        --material C:/tmp/t0btn_smoke.jsonl --out-dir D:/ofc_data/hu/t0_btn_evalfix
    python -m ai.tutor.encode_t0_material --seat btn \\
        --material runs/btnmine3/mine --out-dir D:/ofc_data/hu/t0_btn_evalfix
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ai.tutor.encode_fl14_teacher import context_block
from ai.tutor.encode_fl_material import forward, load_bin, rows_of
from ai.tutor.fl14_allocation_features import allocation_rank_block
from ai.tutor.t0_mine import run_batch
from ai.tutor.t3_second_features import actor_block

D = Path("D:/ofc_data/hu")
REPO = Path(__file__).resolve().parents[2]
OWN_SIZE = 110
OPP_SIZE = 97
FEATURE_SIZE = OWN_SIZE + OPP_SIZE  # 207
SERVE_JOINT_SAMPLES = 200
SERVE_NODE_SEED = "model-rank/t0"
JOINT_ARRANGEMENTS = 32

# `hu_match::pool_of` walks the Rust `all_cards()` order (s, h, d, c; 2..A),
# NOT ai.engine.encoding.ALL_CARDS (h, d, c, s).  The sampler draws by pool
# index, so the pool must be handed to the block binary in the serve order.
RUST_NATURALS = [f"{r}{s}" for s in "shdc" for r in "23456789TJQKA"]


def opening_key_rows(key: str) -> list[list[str]]:
    rows = rows_of(key)
    if sum(len(r) for r in rows) != 5:
        raise AssertionError(f"a T0 opening places five cards: {key}")
    return rows


def serve_pool(own: list[list[str]], opp: list[list[str]]) -> list[str]:
    """The deck `t0_model_scores` judges against: `pool_of(empty, opp, [], hero)`.

    Naturals are removed by name, jokers by count, and the unseen jokers are
    appended as X1.. regardless of which name hero holds -- reproduced here
    because the joint block samples by pool index.
    """
    naturals: set[str] = set()
    jokers = 0
    for card in [c for row in own for c in row] + [c for row in opp for c in row]:
        if card.startswith("X"):
            jokers += 1
        elif card in naturals:
            raise AssertionError(f"card {card} appears twice")
        else:
            naturals.add(card)
    if jokers > 2:
        raise AssertionError(f"{jokers} jokers across the pair; the deck holds two")
    return [c for c in RUST_NATURALS if c not in naturals] + [f"X{i + 1}" for i in range(2 - jokers)]


def rank_root(binary: Path, models: Path, fl_ev: Path, cards: str, seat: int,
              opp_board: str | None, joint: int, topk: int, out: Path) -> list[dict]:
    """Every opening this root's serve can reach, with its evaluator score and
    fence rank -- the audit's own command (`t0_mine.run_batch`), so the score
    column is the one the material's `model_pick` was chosen from."""
    if not out.exists() or not out.stat().st_size:
        return run_batch(binary, models, fl_ev, cards, None, 0, 1, out, joint, topk, seat, opp_board)
    return [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]


def fetch_blocks(binary: Path, fl_ev: Path, requests: list[dict], samples: int,
                 seed: str) -> dict[str, tuple[list[float], list[float]]]:
    """(rowwise 41, joint 8) per request id from `--joint-outlook`.

    One seed for every request: serving encodes all of a node's candidates
    (and the opponent tail) under the node seed, never the candidate's.
    """
    with tempfile.TemporaryDirectory() as tmp:
        in_path, out_path = Path(tmp) / "in.jsonl", Path(tmp) / "out.jsonl"
        with in_path.open("w", encoding="utf-8") as handle:
            for request in requests:
                handle.write(json.dumps(dict(request, samples=samples, seed=seed,
                                             max_arrangements=JOINT_ARRANGEMENTS)) + "\n")
        done = subprocess.run(
            [str(binary), "--input", str(in_path), "--output", str(out_path),
             "--fl-ev-config", str(fl_ev), "--joint-outlook", "--chunk-size", "256"],
            capture_output=True, text=True)
        if done.returncode != 0:
            raise RuntimeError(f"joint-outlook failed: {done.stderr[-600:]}")
        blocks = {}
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                payload = json.loads(line)
                blocks[payload["id"]] = (payload["rowwise_block"], payload["joint_block"])
        return blocks


def board_request(rid: str, rows: list[list[str]], pool: list[str]) -> dict:
    return {"id": rid, "board": {"top": rows[0], "middle": rows[1], "bottom": rows[2]}, "pool": pool}


def encode(binary: Path, fl_ev: Path, pairs: list[tuple[list[list[str]], list[list[str]]]],
           samples: int, seed: str, batch: int) -> np.ndarray:
    """The 207-dim vector for each (own rows, opp rows), in the order given.

    The opponent half is the same for every opening of a root, so it is
    fetched once per distinct (opp board, pool) rather than once per row.
    """
    out = np.empty((len(pairs), FEATURE_SIZE), np.float32)
    for base in range(0, len(pairs), batch):
        chunk = pairs[base:base + batch]
        requests, tails, pools = [], {}, []
        for index, (own, opp) in enumerate(chunk):
            pool = serve_pool(own, opp)
            pools.append(pool)
            requests.append(board_request(f"o{index}", own, pool))
            tail_key = (tuple(tuple(r) for r in opp), tuple(pool))
            if tail_key not in tails:
                tails[tail_key] = f"t{len(tails)}"
                requests.append(board_request(tails[tail_key], opp, pool))
        blocks = fetch_blocks(binary, fl_ev, requests, samples, seed)
        for index, (own, opp) in enumerate(chunk):
            pool = pools[index]
            own_rowwise, own_joint = blocks[f"o{index}"]
            opp_rowwise, opp_joint = blocks[tails[(tuple(tuple(r) for r in opp), tuple(pool))]]
            own_actor, _ = actor_block(own, pool)
            opp_actor, _ = actor_block(opp, pool)
            vector = (own_actor + [float(v) for v in own_rowwise] + [float(v) for v in own_joint]
                      + context_block(pool) + allocation_rank_block(own)
                      + opp_actor + [float(v) for v in opp_rowwise] + [float(v) for v in opp_joint])
            if len(vector) != FEATURE_SIZE:
                raise AssertionError(f"feature size drifted: {len(vector)}")
            out[base + index] = vector
    return out


def load_material(spec: Path) -> list[dict]:
    """One jsonl, a directory of them, or a glob -- fleet shards land as
    `runs/<job>/mine/*.jsonl`.  Ids are checked unique across files."""
    if spec.is_dir():
        files = sorted(spec.glob("*.jsonl"))
    elif spec.exists():
        files = [spec]
    else:
        files = [Path(p) for p in sorted(glob.glob(str(spec)))]
    if not files:
        raise SystemExit(f"no material under {spec}")
    rows, seen = [], set()
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a preempted worker leaves one truncated final line
            if row["id"] in seen:
                raise SystemExit(f"duplicate root id {row['id']} (second copy in {path})")
            seen.add(row["id"])
            rows.append(row)
    print(f"material: {len(rows)} roots from {len(files)} file(s)", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seat", choices=("bb", "btn"), default="btn")
    ap.add_argument("--material", type=Path, required=True,
                    help="a t0_mine jsonl, a directory of them, or a glob")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--models-dir", type=Path, default=D / "models_ship_20260903")
    ap.add_argument("--binary", type=Path,
                    default=REPO / "ai/rust_solver/target/release/t4_first_exact.exe")
    ap.add_argument("--fl-ev-config", type=Path, default=None,
                    help="defaults to <models-dir>/fl_ev.json, the bundle the audit served")
    ap.add_argument("--topk", type=int, default=4,
                    help="ranker fence the audit served at BTN (t0_mine --topk)")
    ap.add_argument("--serve-joint", type=int, default=SERVE_JOINT_SAMPLES,
                    help="--serve-joint-samples the audit ranked with")
    ap.add_argument("--joint-samples", type=int, default=SERVE_JOINT_SAMPLES,
                    help="joint completions per board; 200 = serve spec (parity-exact), "
                         "400 = training spec (the T0-BB design's choice)")
    ap.add_argument("--joint-seed", default=SERVE_NODE_SEED,
                    help="node seed for every board; serving uses model-rank/t0")
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--rank-workers", type=int, default=4)
    args = ap.parse_args()
    seat = 1 if args.seat == "btn" else 0
    fl_ev = args.fl_ev_config or args.models_dir / "fl_ev.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ranks_dir = args.out_dir / "ranks"
    ranks_dir.mkdir(exist_ok=True)
    started = time.time()

    material = load_material(args.material)
    for row in material:
        got = row.get("seat", "bb")
        if got != args.seat:
            raise SystemExit(f"root {row['id']} is seat {got}, run asked for {args.seat}")
        if seat == 1 and not row.get("opp_board"):
            raise SystemExit(f"root {row['id']}: a BTN root needs opp_board")

    # Ranking is one short process per root; the binary threads the 232
    # encodings, so a few side by side fill the box.
    def ranked(row):
        wid = row["id"].replace("/", "_")
        return row["id"], rank_root(args.binary, args.models_dir, fl_ev, row["cards"], seat,
                                    row.get("opp_board"), args.serve_joint, args.topk,
                                    ranks_dir / f"{wid}.jsonl")

    with ThreadPoolExecutor(max_workers=args.rank_workers) as pool:
        ranks = dict(pool.map(ranked, material))
    print(f"ranked {len(ranks)} roots in {time.time() - started:.0f}s", flush=True)

    # BB serves through policy.bin's top-8 when the bundle carries it; BTN
    # never does (Rust leaves policy_rank null) and the ranker's top-K selects.
    policy_fence = seat == 0 and (args.models_dir / "policy.bin").exists()
    fence_key, fence_k = ("policy_rank", 8) if policy_fence else ("ranker_rank", args.topk)

    pairs, meta, requests = [], [], []
    for row in material:
        sel = row["sel_means"]
        precise = "ci" in row
        opp_rows = rows_of(row["opp_board"]) if seat == 1 else [[], [], []]
        entries = ranks[row["id"]]
        if len(entries) != 232:
            raise AssertionError(f"{row['id']}: {len(entries)} openings ranked, expected 232")
        keys = {e["key"] for e in entries}
        for k in (row["model_pick"], row["ref_pick"], *sel):
            if k not in keys:
                raise AssertionError(f"{row['id']}: audited key {k} is not an opening the serve ranked")
        for entry in entries:
            key = entry["key"]
            is_serving = key == row["model_pick"]
            is_ref = key == row["ref_pick"]
            fence_rank = entry.get(fence_key)
            record = {
                "id": f"{row['id']}/{key}", "root": row["id"], "key": key,
                "seat": args.seat, "opp_board": row.get("opp_board"),
                "mean": sel.get(key), "in_sel": key in sel,
                "scored": precise, "escalated": bool(row.get("escalated", False)),
                "is_serving": is_serving, "is_ref": is_ref,
                "verdict": row["verdict"],
                "own_rank": entry["rank"], "own_score": entry["score"],
                "fence_rank": fence_rank, "in_fence": fence_rank is not None and fence_rank <= fence_k,
                "ranker_rank": entry.get("ranker_rank"), "policy_rank": entry.get("policy_rank"),
            }
            if precise and (is_serving or is_ref):
                record["margin"] = row["margin"]
                record["ci"] = row["ci"]
            own_rows = opening_key_rows(key)
            pairs.append((own_rows, opp_rows))
            meta.append(record)
            requests.append({"id": record["id"],
                             "board": {"top": own_rows[0], "middle": own_rows[1], "bottom": own_rows[2]},
                             "opp_board": {"top": opp_rows[0], "middle": opp_rows[1], "bottom": opp_rows[2]},
                             "dead": [], "seed": args.joint_seed})
    audited = sum(m["in_sel"] for m in meta)
    print(f"boards: {len(pairs)} ({audited} audited, fence {fence_key}<={fence_k})", flush=True)
    # The `--hu-encode` request shape, one per row, so the Rust encoder can be
    # run over the same boards (at its own 400/id spec) whenever wanted.
    with open(args.out_dir / "enc_requests.jsonl", "w", encoding="utf-8") as handle:
        for request in requests:
            handle.write(json.dumps(request, ensure_ascii=False) + "\n")

    x = encode(args.binary, fl_ev, pairs, args.joint_samples, args.joint_seed, args.batch)
    print(f"encoded in {time.time() - started:.0f}s "
          f"(joint {args.joint_samples}x{JOINT_ARRANGEMENTS}, seed {args.joint_seed})", flush=True)

    # Parity: the vectors must reproduce the score the serving path published
    # for the same placement.  Checked on every row, not a sample -- it costs
    # one matmul and it is the only evidence that this file and
    # `hu_encode::board_blocks` agree.
    bin_path = args.models_dir / "hu" / f"t0_{args.seat}.bin"
    mean, std, mats = load_bin(bin_path)
    if len(mean) != FEATURE_SIZE:
        raise SystemExit(f"{bin_path} reads {len(mean)} dims, this encoder writes {FEATURE_SIZE}")
    py = forward(mats, (x - mean) / std)
    rust = np.asarray([m["own_score"] for m in meta], np.float32)
    delta = np.abs(py - rust)
    # Two argmaxes are compared, both against the RUST score column of the
    # same run, never against `model_pick`: at Button `model_pick` is the move
    # the champion played in the traced hand, under that hand's own joint
    # sample stream, and t0_mine.py notes the audit's fenced argmax reproduces
    # it only 39/50.  Whether it matches is reported, not gated.
    tally = {scope: dict(agree=0, tied=0, split=0) for scope in ("all", "fence")}
    served_match = at = 0
    for row in material:
        n = len(ranks[row["id"]])
        window = py[at:at + n]
        rust_win = rust[at:at + n]
        keys = [m["key"] for m in meta[at:at + n]]
        inside = [i for i in range(n) if meta[at + i]["in_fence"]] or list(range(n))
        for scope, indices in (("all", list(range(n))), ("fence", inside)):
            pick = max(indices, key=lambda i: float(window[i]))
            rust_pick = max(indices, key=lambda i: float(rust_win[i]))
            if pick == rust_pick:
                tally[scope]["agree"] += 1
            elif abs(float(window[pick]) - float(window[rust_pick])) <= 1e-4:
                # Tie-equal scores: Rust breaks by enumeration order, numpy by
                # array order.  What must match is the SCORE.
                tally[scope]["tied"] += 1
            else:
                tally[scope]["split"] += 1
        if keys[max(inside, key=lambda i: float(rust_win[i]))] == row["model_pick"]:
            served_match += 1
        at += n
    if at != len(meta):
        raise AssertionError(f"parity walked {at} of {len(meta)} rows")
    agree, tied, split = (tally["all"][k] for k in ("agree", "tied", "split"))
    fence_agree, fence_tied, fence_split = (tally["fence"][k] for k in ("agree", "tied", "split"))
    print(f"PARITY max|py-rust| {delta.max():.3e}  mean {delta.mean():.3e}", flush=True)
    print(f"PARITY argmax(all 232) py==rust {agree}/{len(material)} roots "
          f"(+{tied} tie-equal, {split} genuine disagreements); "
          f"fenced argmax py==rust {fence_agree}/{len(material)} "
          f"(+{fence_tied} tie-equal, {fence_split} genuine)", flush=True)
    print(f"INFO audit's fenced argmax == model_pick (the move played) "
          f"{served_match}/{len(material)}", flush=True)
    split += fence_split
    # Written before the gate: a parity failure is something to diagnose, and
    # re-encoding just to look at it is a waste.  The exit code, not the
    # presence of the file, is the contract.
    np.savez_compressed(args.out_dir / "enc_pairs.npz", x=x,
                        own_score=rust, py_score=py.astype(np.float32))
    with open(args.out_dir / "pairs_meta.jsonl", "w", encoding="utf-8") as handle:
        for record in meta:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (args.out_dir / "manifest.json").write_text(json.dumps(dict(
        seat=args.seat, feature_size=FEATURE_SIZE, joint_samples=args.joint_samples,
        joint_seed=args.joint_seed, serve_joint=args.serve_joint, fence=[fence_key, fence_k],
        bin=str(bin_path), roots=len(material), rows=len(meta), audited=audited,
        parity_max_abs=float(delta.max()), parity_mean_abs=float(delta.mean()),
        argmax_agree=agree, argmax_tied=tied, argmax_split=split - fence_split,
        fence_argmax_agree=fence_agree, fence_argmax_tied=fence_tied,
        fence_argmax_split=fence_split, served_matches_fenced_argmax=served_match),
        indent=2), encoding="utf-8")
    print(f"wrote {args.out_dir/'enc_pairs.npz'} x{x.shape}, pairs_meta.jsonl, "
          f"enc_requests.jsonl ({time.time() - started:.0f}s total)", flush=True)
    serve_spec = args.joint_samples == args.serve_joint and args.joint_seed == SERVE_NODE_SEED
    if serve_spec:
        if delta.max() > 1e-3 or split:
            raise SystemExit("PARITY FAILED -- the Python encoder is not the serving encoder")
    else:
        # Off the serve spec the joint dims differ by sampling, so a score gap
        # is expected; the check that still binds is the fenced decision.
        print(f"NOTE joint spec {args.joint_samples}/{args.joint_seed} is not the served "
              f"{args.serve_joint}/{SERVE_NODE_SEED}; parity is reported, not gated", flush=True)
        if fence_agree < len(material):
            print(f"WARNING fenced argmax moved on {len(material) - fence_agree} roots under "
                  f"the off-spec joint", flush=True)


if __name__ == "__main__":
    main()
