"""Export the trained T0-BB policy net as a flat little-endian image.

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
    input_dim   u32            = 54
    mean        f32[54]        (zeros)
    std         f32[54]        (ones)
    per layer:  inputs u32, outputs u32, weight f32[in*out] (row-major
                [output][input]), bias f32[out]

The exported bytes are read back with plain numpy and compared against torch
on random probes, so a transposed weight or a truncated tail is caught here
rather than by a match that quietly serves a different net.

Usage:
    python -m ai.tutor.export_t0_policy \
        --model D:/ofc_data/hu/t0_policy_v1/policy_best.pt \
        --out   D:/ofc_data/hu/t0_policy_v1/policy.bin
"""
from __future__ import annotations

import argparse
import hashlib
import struct
from pathlib import Path

import numpy as np
import torch

from ai.tutor.train_t0_policy import MASK, Policy

MAGIC = b"T4F1"
VERSION = 1


def export(model_path: Path, out_path: Path) -> dict:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    model = Policy()
    model.load_state_dict(state)
    model.eval()

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
    parser.add_argument("--probes", type=int, default=512)
    args = parser.parse_args()
    info = export(args.model, args.out)

    # Probe on real five-hot vectors, not gaussian noise: the net only ever
    # sees five ones, and a bug that only shows up off that manifold is a bug
    # nobody serves.
    rng = np.random.default_rng(20260828)
    probe = np.zeros((args.probes, info["input_dim"]), np.float32)
    for row in range(args.probes):
        probe[row, rng.choice(info["input_dim"], 5, replace=False)] = 1.0

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    model = Policy()
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()
    with torch.no_grad():
        # `Policy.forward` masks; the image cannot, so compare the raw stack
        # and check the mask separately where it is actually applied.
        reference = model.net(torch.from_numpy(probe)).numpy()
    exported = forward_image(args.out, probe)
    gap = float(np.abs(reference - exported).max())
    ranked_ref = np.argsort(-np.where(MASK, reference, -np.inf), axis=1)
    ranked_exp = np.argsort(-np.where(MASK, exported, -np.inf), axis=1)
    order_gap = int((ranked_ref != ranked_exp).sum())
    print(info)
    print(f"probe: max |torch - numpy| = {gap:.3e} over {args.probes} probes, "
          f"legal-action order mismatches {order_gap}")
    if gap > 1e-4 or order_gap:
        raise SystemExit("export does not reproduce the torch net")


if __name__ == "__main__":
    main()
