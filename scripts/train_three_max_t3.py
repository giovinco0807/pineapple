"""Distil the 3-max T3-BTN teacher into a ranking model.

    python scripts/train_three_max_t3.py --corpus D:/ofc_data/.../corpus.jsonl \
        --output D:/ofc_data/three_max_t3_btn/model

Protocol is the heads-up one, which was frozen after the cascade measured it:
120 epochs, Adam 1e-3 cosined to zero, batch 512 ROOTS (not actions), inputs
standardised on the training split, and the epoch with the lowest held-out
regret is the one kept.

The loss splits the label into the two parts that matter differently.  Within a
root, only the SPREAD of the action values decides which move is played, and
that is what the ranking term fits.  The root's mean value decides nothing
about this decision but is what an upstream street will later read out of this
model, so it is fitted too, at ``--reg-weight``.

The metric is regret -- how much label value the model's pick gives up against
the root's best action -- not top-1 accuracy.  Heads-up learned this twice: a
model can be right 68% of the time and still be worthless, or wrong most of the
time on near-ties and be fine.  A regret number is only meaningful next to the
teacher's own noise floor, which ``measure_floor`` in this file computes by
relabelling the same roots from a disjoint seed stream.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from ofc_regular.three_max import features as _feat  # noqa: E402
from ofc_regular.three_max.features import encode_record_action  # noqa: E402
FEATURE_SIZE = _feat.FEATURE_SIZE

MODEL_SCHEMA = "regular_ofc_3max_t3_btn_model_v1"
HOLDOUT_SPLIT_CODE = 230


def stable_holdout(seed: int, *, fraction: float) -> bool:
    """Deterministic split that does not move when the corpus is extended."""
    digest = hashlib.sha256(f"{HOLDOUT_SPLIT_CODE}:{seed}".encode("ascii")).digest()
    return (int.from_bytes(digest[:4], "big") % 10_000) < fraction * 10_000


def load_corpus(path: Path, *, limit: int | None = None) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
            if limit and len(records) >= limit:
                break
    return records


def encode_corpus(records: list[dict]) -> tuple[np.ndarray, np.ndarray, list[slice]]:
    """Feature matrix, label vector, and one slice per root."""
    features: list[list[float]] = []
    labels: list[float] = []
    groups: list[slice] = []
    for record in records:
        start = len(features)
        for index in range(len(record["actions"])):
            features.append(list(encode_record_action(record, index).features))
            labels.append(record["actions"][index]["ev"])
        groups.append(slice(start, len(features)))
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
        groups,
    )


def load_encoded(path: Path) -> tuple[np.ndarray, np.ndarray, list[slice], np.ndarray]:
    """Read a cache written by scripts/encode_three_max_t3.py."""
    payload = np.load(path, allow_pickle=False)
    meta = json.loads(str(payload["meta"][0]))
    global FEATURE_SIZE
    FEATURE_SIZE = int(meta["feature_size"])
    features = payload["features"]
    labels = payload["labels"]
    widths = payload["widths"]
    seeds = payload["seeds"]
    groups: list[slice] = []
    offset = 0
    for width in widths:
        groups.append(slice(offset, offset + int(width)))
        offset += int(width)
    return features, labels, groups, seeds


class Ranker(nn.Module):
    def __init__(self, input_dim: int = FEATURE_SIZE, width: float = 1.0) -> None:
        super().__init__()
        sizes = [max(4, int(round(n * width))) for n in (256, 128, 64)]
        self.body = nn.Sequential(
            nn.Linear(input_dim, sizes[0]),
            nn.ReLU(),
            nn.Linear(sizes[0], sizes[1]),
            nn.ReLU(),
            nn.Linear(sizes[1], sizes[2]),
            nn.ReLU(),
            nn.Linear(sizes[2], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x).squeeze(-1)


def _pack(
    features: np.ndarray, labels: np.ndarray, groups: list[slice], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad every root to the true maximum fan width and carry a mask.

    Truncating instead of packing to the true width silently drops rows -- the
    heads-up trainer lost about 3% of its widest street that way.

    Each root's actions are also PERMUTED here.  The corpus stores them sorted
    by label EV, so index 0 is the teacher's answer -- and argmax over tied
    scores returns index 0.  A model with no input at all therefore measured
    regret 0.0000 before this permutation existed, and every regret reported
    from this script was inflated toward the answer by the same leak.  The
    permutation is seeded per root, so it is deterministic without being
    informative.
    """
    width = max(group.stop - group.start for group in groups)
    packed = np.zeros((len(groups), width, features.shape[1]), dtype=np.float32)
    packed_labels = np.zeros((len(groups), width), dtype=np.float32)
    mask = np.zeros((len(groups), width), dtype=np.float32)
    for row, group in enumerate(groups):
        size = group.stop - group.start
        order = np.random.default_rng(0x5EED + row).permutation(size)
        packed[row, :size] = features[group][order]
        packed_labels[row, :size] = labels[group][order]
        mask[row, :size] = 1.0
    return (
        torch.from_numpy(packed).to(device),
        torch.from_numpy(packed_labels).to(device),
        torch.from_numpy(mask).to(device),
    )


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum(dim=1) / mask.sum(dim=1)


def evaluate(
    model: Ranker,
    packed: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        scores = model(packed)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        picks = scores.argmax(dim=1)
        chosen = labels.gather(1, picks.unsqueeze(1)).squeeze(1)
        best = labels.masked_fill(mask == 0, float("-inf")).max(dim=1).values
        regret = (best - chosen)

        order = scores.argsort(dim=1, descending=True)
        label_best = labels.masked_fill(mask == 0, float("-inf")).argmax(dim=1)
        top1 = (order[:, 0] == label_best).float().mean().item()
        top3 = (
            (order[:, :3] == label_best.unsqueeze(1)).any(dim=1).float().mean().item()
        )
    return {
        "regret": regret.mean().item(),
        "regret_p95": regret.quantile(0.95).item(),
        "top1": top1,
        "top3": top3,
    }


def _gather(
    features: np.ndarray, labels: np.ndarray, groups: list[slice], keep: list[int]
) -> tuple[np.ndarray, np.ndarray, list[slice]]:
    parts_x = [features[groups[i]] for i in keep]
    parts_y = [labels[groups[i]] for i in keep]
    out_groups: list[slice] = []
    offset = 0
    for part in parts_x:
        out_groups.append(slice(offset, offset + len(part)))
        offset += len(part)
    return np.concatenate(parts_x), np.concatenate(parts_y), out_groups


def train(
    *,
    corpus: Path,
    encoded: Path | None,
    output: Path,
    epochs: int,
    batch_roots: int,
    learning_rate: float,
    reg_weight: float,
    holdout_fraction: float,
    limit: int | None,
    device_name: str,
    width: float = 1.0,
    mask_blocks: tuple[int, int] | None = None,
    init_seed: int = 997_990_013,
) -> dict:
    started = time.time()
    if encoded is not None:
        features, labels, groups, seeds = load_encoded(encoded)
        keep_train = [i for i, s in enumerate(seeds) if not stable_holdout(int(s), fraction=holdout_fraction)]
        keep_hold = [i for i, s in enumerate(seeds) if stable_holdout(int(s), fraction=holdout_fraction)]
        # --limit trims TRAINING roots only.  Shrinking the holdout too would
        # change the yardstick between ablation arms and make them
        # incomparable, which is exactly the mistake this guards against.
        if limit:
            keep_train = keep_train[:limit]
        train_x, train_y, train_groups = _gather(features, labels, groups, keep_train)
        holdout_x, holdout_y, holdout_groups = _gather(features, labels, groups, keep_hold)
        n_train, n_hold = len(keep_train), len(keep_hold)
    else:
        records = load_corpus(corpus, limit=limit)
        train_records = [r for r in records if not stable_holdout(r["seed"], fraction=holdout_fraction)]
        holdout_records = [r for r in records if stable_holdout(r["seed"], fraction=holdout_fraction)]
        train_x, train_y, train_groups = encode_corpus(train_records)
        holdout_x, holdout_y, holdout_groups = encode_corpus(holdout_records)
        n_train, n_hold = len(train_records), len(holdout_records)
    if not holdout_groups:
        raise ValueError("holdout split is empty")
    print(
        f"roots: {n_train} train / {n_hold} holdout; "
        f"{len(train_y)} + {len(holdout_y)} actions in {time.time()-started:.1f}s",
        flush=True,
    )

    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-6] = 1.0
    train_x = (train_x - mean) / std
    holdout_x = (holdout_x - mean) / std

    # Ablate AFTER standardising and BEFORE packing.  Zeroing a standardised
    # column removes its signal while leaving it at the training mean, so the
    # network sees "no information" rather than an out-of-distribution value.
    # Placing this after _pack silently does nothing -- the packed tensors are
    # already copies -- which is how an earlier version of this script produced
    # a whole ablation table that was really just seed noise.
    if mask_blocks is not None:
        start, stop = mask_blocks
        train_x[:, start:stop] = 0.0
        holdout_x[:, start:stop] = 0.0

    device = torch.device(device_name)
    packed, packed_y, mask = _pack(train_x, train_y, train_groups, device)
    hold_packed, hold_y, hold_mask = _pack(holdout_x, holdout_y, holdout_groups, device)

    torch.manual_seed(init_seed)
    model = Ranker(input_dim=train_x.shape[1], width=width).to(device)
    params = sum(p.numel() for p in model.parameters())
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    label_mean = _masked_mean(packed_y, mask).unsqueeze(1)
    centred_labels = (packed_y - label_mean) * mask

    best = {"regret": float("inf")}
    best_state = None
    generator = torch.Generator(device="cpu").manual_seed(997_990_013)
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(packed.shape[0], generator=generator)
        for start in range(0, len(order), batch_roots):
            index = order[start : start + batch_roots].to(device)
            batch, batch_labels, batch_mask = packed[index], centred_labels[index], mask[index]
            batch_offset = label_mean[index]

            scores = model(batch)
            score_mean = _masked_mean(scores, batch_mask).unsqueeze(1)
            centred_scores = (scores - score_mean) * batch_mask

            ranking = ((centred_scores - batch_labels) ** 2 * batch_mask).sum() / batch_mask.sum()
            level = ((score_mean - batch_offset) ** 2).mean()
            loss = ranking + reg_weight * level

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        schedule.step()

        metrics = evaluate(model, hold_packed, hold_y, hold_mask)
        if metrics["regret"] < best["regret"]:
            best = {**metrics, "epoch": epoch}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 10 == 0:
            print(
                f"  epoch {epoch+1:3d}  holdout regret {metrics['regret']:.4f}  "
                f"top1 {metrics['top1']:.3f}  top3 {metrics['top3']:.3f}",
                flush=True,
            )

    assert best_state is not None
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": MODEL_SCHEMA,
            "input_dim": FEATURE_SIZE,
            "mean": mean,
            "std": std,
            "state_dict": best_state,
            "metrics": best,
        },
        output.with_suffix(".pt"),
    )
    report = {
        "schema": MODEL_SCHEMA,
        "corpus": str(corpus),
        "encoded_cache": str(encoded) if encoded else None,
        "roots_train": n_train,
        "roots_holdout": n_hold,
        "actions_train": int(len(train_y)),
        "input_dim": FEATURE_SIZE,
        "width": width,
        "params": params,
        "masked_block": list(mask_blocks) if mask_blocks else None,
        "init_seed": init_seed,
        "epochs": epochs,
        "batch_roots": batch_roots,
        "learning_rate": learning_rate,
        "reg_weight": reg_weight,
        "holdout_fraction": holdout_fraction,
        "holdout_split_code": HOLDOUT_SPLIT_CODE,
        "best": best,
        "wall_seconds": round(time.time() - started, 1),
    }
    output.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--encoded", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-roots", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--reg-weight", type=float, default=0.1)
    parser.add_argument("--width", type=float, default=1.0)
    parser.add_argument("--init-seed", type=int, default=997_990_013)
    parser.add_argument("--mask-block", type=str, default=None, help="start:stop feature columns to zero")
    parser.add_argument("--holdout-fraction", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None,
                        help="cap TRAINING roots; the holdout is never trimmed")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    if args.corpus is None and args.encoded is None:
        parser.error("one of --corpus or --encoded is required")
    report = train(
        corpus=args.corpus,
        encoded=args.encoded,
        output=args.output,
        epochs=args.epochs,
        batch_roots=args.batch_roots,
        learning_rate=args.learning_rate,
        reg_weight=args.reg_weight,
        holdout_fraction=args.holdout_fraction,
        limit=args.limit,
        device_name=args.device,
        width=args.width,
        init_seed=args.init_seed,
        mask_blocks=(
            tuple(int(v) for v in args.mask_block.split(':')) if args.mask_block else None
        ),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
