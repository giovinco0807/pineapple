"""Export a trained T0 policy net (BB, 54-in, or BTN, 213-in) as a flat
little-endian image.

Same T4F1 container as `ai/tutor/export_t4_first_evaluator.py`, because the
Rust side already has one loader and a second format would be a second place
for an encoder to drift away from its weights.  Two differences, both benign:

  * the policy net standardises nothing, so `mean` is a zero vector and `std`
    a ones vector -- the loader's affine becomes the identity rather than a
    special case;
  * the last layer has 243 outputs rather than one, which the reader must be
    told to allow (`Model::load_wide` on the Rust side).

    magic       b"T4F1"        4 bytes
    version     u32            = 1
    layers      u32
    input_dim   u32            = 54 (BB, `--seat bb`) or 213 (BTN, `--seat btn`)
    mean        f32[input_dim] (zeros)
    std         f32[input_dim] (ones)
    per layer:  inputs u32, outputs u32, weight f32[in*out] (row-major
                [output][input]), bias f32[out]

The input width is whatever the checkpoint's first layer says; the seat only
picks which training module rebuilds the net and how the probe states are
made.  The exported bytes are read back with plain numpy and compared
against torch on probes -- five-hot vectors for BB, real (hero, opponent
board) states from the request files for BTN, because the net only ever
sees that manifold and a bug that shows up off it is a bug nobody serves --
so a transposed weight or a truncated tail is caught here rather than by a
match that quietly serves a different net.

Usage:
    python -m ai.tutor.export_t0_policy \
        --model D:/ofc_data/hu/t0_policy_v1/policy_best.pt \
        --out   D:/ofc_data/hu/t0_policy_v1/policy.bin
    python -m ai.tutor.export_t0_policy --seat btn \
        --model D:/ofc_data/hu/t0_btn_policy_s1/policy_best.pt \
        --out   D:/ofc_data/hu/t0_btn_policy_s1/policy.bin
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import torch

from ai.tutor.train_t0_policy import MASK, Policy
from ai.tutor.train_t0_btn_policy import encode_state, policy_from_state, rename_hero_jokers

MAGIC = b"T4F1"
VERSION = 1


def load_model(model_path: Path, seat: str):
    """The net the checkpoint was trained as, rebuilt by its own module."""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    if seat == "bb":
        model = Policy()
    elif seat == "btn":
        model = policy_from_state(state)
    else:
        raise SystemExit(f"unknown seat {seat!r}")
    model.load_state_dict(state)
    model.eval()
    return model


def export(model_path: Path, out_path: Path, seat: str = "bb") -> dict:
    model = load_model(model_path, seat)

    linears = [layer for layer in model.net if isinstance(layer, torch.nn.Linear)]
    input_dim = linears[0].weight.shape[1]
    payload = bytearray()
    payload += MAGIC
    payload += struct.pack("<I", VERSION)
    payload += struct.pack("<I", len(linears))
    payload += struct.pack("<I", int(input_dim))
    # No standardisation in training, so the loader's affine is the identity.
    payload += np.zeros(input_dim, np.float32).tobytes()
    payload += np.ones(input_dim, np.float32).tobytes()
    for linear in linears:
        weight = linear.weight.detach().to(torch.float32).contiguous()
        bias = linear.bias.detach().to(torch.float32).contiguous()
        payload += struct.pack("<II", weight.shape[1], weight.shape[0])
        payload += weight.numpy().tobytes()
        payload += bias.numpy().tobytes()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(bytes(payload))
    return {
        "path": str(out_path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "input_dim": int(input_dim),
        "layers": [(int(l.weight.shape[1]), int(l.weight.shape[0])) for l in linears],
    }


def read_image(path: Path):
    """Re-read the exported bytes with numpy alone: mean, inverse std, layers."""
    blob = path.read_bytes()
    assert blob[:4] == MAGIC, "bad magic"
    offset = 4
    version, layer_count, input_dim = struct.unpack_from("<III", blob, offset)
    assert version == VERSION, f"unexpected version {version}"
    offset += 12
    mean = np.frombuffer(blob, np.float32, input_dim, offset).copy()
    offset += 4 * input_dim
    std = np.frombuffer(blob, np.float32, input_dim, offset).copy()
    offset += 4 * input_dim
    layers = []
    for _ in range(layer_count):
        ins, outs = struct.unpack_from("<II", blob, offset)
        offset += 8
        weight = np.frombuffer(blob, np.float32, ins * outs, offset).copy().reshape(outs, ins)
        offset += 4 * ins * outs
        bias = np.frombuffer(blob, np.float32, outs, offset).copy()
        offset += 4 * outs
        layers.append((weight, bias))
    assert offset == len(blob), f"{len(blob) - offset} trailing bytes"
    return mean, std, layers


def forward_image(path: Path, x: np.ndarray) -> np.ndarray:
    """Run the exported image the way the Rust loader runs it."""
    mean, std, layers = read_image(path)
    h = (x - mean) / std
    for index, (weight, bias) in enumerate(layers):
        h = h @ weight.T + bias
        if index + 1 < len(layers):
            h = np.maximum(h, 0.0)
    return h


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path,
        default=Path("D:/ofc_data/hu/t0_policy_v1/policy_best.pt"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seat", choices=("bb", "btn"), default="bb")
    parser.add_argument(
        "--requests", type=Path,
        default=Path("D:/ofc_data/hu/onpol2_requests/t0_btn_lap2.jsonl"),
        help="BTN only: request rows (draw + opp_board) the probe states are taken from",
    )
    parser.add_argument("--probes", type=int, default=512)
    args = parser.parse_args()
    info = export(args.model, args.out, args.seat)

    rng = np.random.default_rng(20260828)
    if args.seat == "bb":
        # Probe on real five-hot vectors, not gaussian noise: the net only
        # ever sees five ones, and a bug that only shows up off that manifold
        # is a bug nobody serves.
        probe = np.zeros((args.probes, info["input_dim"]), np.float32)
        for row in range(args.probes):
            probe[row, rng.choice(info["input_dim"], 5, replace=False)] = 1.0
    else:
        # Real (hero, opponent board) states through the training encoder --
        # jokers on both sides included, as the request files carry them.
        rows = [json.loads(line) for line in args.requests.open(encoding="utf-8")]
        picked = rng.choice(len(rows), min(args.probes, len(rows)), replace=False)
        probe = np.stack([
            encode_state(rename_hero_jokers(rows[i]["draw"]), rows[i]["opp_board"])[0]
            for i in picked
        ])
        assert probe.shape[1] == info["input_dim"], (probe.shape, info["input_dim"])
        jokers = int(sum(1 for i in picked if any(
            c.startswith("X") for c in rows[i]["draw"] + sum(rows[i]["opp_board"], []))))
        print(f"probe: {len(probe)} real BTN states from {args.requests} ({jokers} with a joker)")

    model = load_model(args.model, args.seat)
    with torch.no_grad():
        # `Policy.forward` masks; the image cannot, so compare the raw stack
        # and check the mask separately where it is actually applied.
        reference = model.net(torch.from_numpy(probe)).numpy()
    exported = forward_image(args.out, probe)
    gap = float(np.abs(reference - exported).max())
    ranked_ref = np.argsort(-np.where(MASK, reference, -np.inf), axis=1)
    ranked_exp = np.argsort(-np.where(MASK, exported, -np.inf), axis=1)
    # A rank position naming different actions on the two sides is an export
    # defect only when the two actions' logits are actually apart.  Over
    # 512 x 232 outputs exact f32 ties between two actions occur at O(1)
    # counts, and a different accumulation order (numpy here, Rust's f32
    # loop in serving) resolves such a tie either way -- so the flips that
    # count are those beyond the same tolerance the values are held to.
    flips = ranked_ref != ranked_exp
    pair_gap = np.abs(np.take_along_axis(reference, ranked_ref, 1)
                      - np.take_along_axis(reference, ranked_exp, 1))
    order_gap = int((flips & (pair_gap > 1e-4)).sum())
    tie_flips = int((flips & (pair_gap <= 1e-4)).sum())
    print(info)
    print(f"probe: max |torch - numpy| = {gap:.3e} over {len(probe)} probes, "
          f"legal-action order mismatches {order_gap} "
          f"(plus {tie_flips} rank flips between logits tied within 1e-4)")
    if gap > 1e-4 or order_gap:
        raise SystemExit("export does not reproduce the torch net")


if __name__ == "__main__":
    main()
