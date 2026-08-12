"""Play example hands with the vs-FL chain and show every placement.

T0..T3 place by each street's learned evaluator (argmax over candidates,
features exactly as in the teachers), T4 places by exact library scoring --
the same chain the teachers' playouts use, shown one street at a time.

Usage:
    python -m ai.tutor.show_vs_fl_chain_hands --hands 3 --seed 555
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import tempfile
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

import ai.tutor.exact_late as exact_late
from ai.engine.action_space import get_turn_actions
from ai.engine.encoding import Board
from ai.tutor.fl_ev_table import FL_EV
from ai.tutor.generate_t0_vs_fl_teacher import rows_of_action_key
from ai.tutor.generate_t2_vs_fl_teacher import encode_t2_action
from ai.tutor.generate_t3_vs_fl_teacher import encode_t3_action
from ai.tutor.t2_policy_label_experiment import ALL_CARDS
from ai.engine.encoding import ALL_CARDS as ALL_CARDS_FULL
from ai.tutor.t4_vs_fl import FlLibrary, hero_terminal, score_against_library, seen_mask
from ai.tutor.train_t4_first_evaluator import T4FirstEvaluator

SOLVER = Path(
    "C:/Users/Owner/AppData/Local/Temp/claude/"
    "C--Users-Owner--gemini-antigravity-scratch-ofc-pineapple/"
    "330b2796-f08e-4d44-8104-95364f5124ba/scratchpad/t4fe_target/release/t4_first_exact.exe"
)
LIBRARY_DIR = "D:/ofc_data/fl_library_14_v3"
MODEL_SETS = {
    "v1": {
        "t0": "D:/ofc_data/t0_vs_fl_model_v1/evaluator_best.pt",
        "t1": "D:/ofc_data/t1_vs_fl_model_v1/evaluator_best.pt",
        "t2": "D:/ofc_data/t2_vs_fl_model_v1/evaluator_best.pt",
        "t3": "D:/ofc_data/t3_vs_fl_model_v2/evaluator_best.pt",
    },
    "v2": {
        "t0": "D:/ofc_data/t0_vs_fl_model_v1/evaluator_best.pt",  # v2 pending
        "t1": "D:/ofc_data/t1_vs_fl_model_v2/evaluator_best.pt",
        "t2": "D:/ofc_data/t2_vs_fl_model_v2/evaluator_best.pt",
        "t3": "D:/ofc_data/t3_vs_fl_model_v2/evaluator_best.pt",
    },
}


class Net:
    def __init__(self, path: str) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.model = T4FirstEvaluator(
            checkpoint["input_dim"], tuple(checkpoint["hidden"])
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.mean = checkpoint["input_mean"].numpy().astype(np.float32)
        self.std = checkpoint["input_std"].numpy().astype(np.float32)

    @property
    def input_dim(self) -> int:
        return int(self.mean.shape[0])

    def predict(self, rows: list[list[float]]) -> np.ndarray:
        x = (np.asarray(rows, dtype=np.float32) - self.mean) / self.std
        with torch.no_grad():
            return self.model(torch.from_numpy(x)).squeeze(-1).numpy()


def run_labeler(args: list[str], requests: list[dict]) -> list[dict]:
    with tempfile.TemporaryDirectory() as tmp:
        in_path = Path(tmp) / "in.jsonl"
        out_path = Path(tmp) / "out.jsonl"
        in_path.write_text(
            "".join(json.dumps(r) + "\n" for r in requests), encoding="utf-8"
        )
        subprocess.run(
            [str(SOLVER), "--input", str(in_path), "--output", str(out_path),
             "--fl-ev-config", "ai/config/fl_ev.json", *args, "--chunk-size", "4"],
            check=True, capture_output=True,
        )
        return [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def show(rows, draw=None, discard=None) -> str:
    parts = [
        f"  top: {' '.join(rows[0]) or '--'}",
        f"  mid: {' '.join(rows[1]) or '--'}",
        f"  bot: {' '.join(rows[2]) or '--'}",
    ]
    if draw is not None:
        parts.append(f"  (draw {' '.join(draw)} / discard {discard})")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hands", type=int, default=3)
    parser.add_argument("--seed", type=int, default=555)
    parser.add_argument("--models", choices=["v1", "v2"], default="v2")
    args = parser.parse_args()

    nets = {name: Net(path) for name, path in MODEL_SETS[args.models].items()}
    library = FlLibrary(Path(LIBRARY_DIR))
    models_bin = [
        "--t0-t1-model", "D:/ofc_data/t1_vs_fl_model_v1/evaluator.bin",
        "--t1-t2-model", "D:/ofc_data/t2_vs_fl_model_v1/evaluator.bin",
        "--t2-t3-model", "D:/ofc_data/t3_vs_fl_model_v1/evaluator.bin",
    ]

    for hand_index in range(args.hands):
        rng = random.Random(args.seed + hand_index)
        deck = ALL_CARDS[:]
        rng.shuffle(deck)
        opp_count = rng.choices([14, 15, 16, 17], weights=[6, 2, 1, 1])[0]
        dealt, cursor = deck[:5], 5
        print(f"\n{'='*58}\nハンド{hand_index+1}  (相手FL {opp_count}枚)")
        print(f"配布5枚: {' '.join(dealt)}")

        # ---- T0: rowwise from the labeler, argmax by the T0 net ----
        response = run_labeler(
            ["--t0-vs-fl-library", LIBRARY_DIR, *models_bin],
            [{"id": "d", "cards": dealt, "opp_count": opp_count,
              "t1_samples": 1, "t2_samples": 1, "t3_samples": 1, "t4_draw_sample": 1}],
        )[0]
        actions = response["actions"]
        vectors = [
            encode_t2_action(rows_of_action_key(a["action_key"]), [], opp_count,
                             a["own_rowwise_block"])
            for a in actions
        ]
        scores = nets["t0"].predict(vectors)
        order = np.argsort(-scores)
        rows = [list(r) for r in rows_of_action_key(actions[order[0]]["action_key"])]
        print(f"T0配置 (モデル値 {scores[order[0]]:+.2f}):")
        print(show(rows))
        print("  次点:", " / ".join(
            f"{actions[i]['action_key']} ({scores[i]:+.2f})" for i in order[1:3]
        ))

        dead: list[str] = []
        # ---- T1, T2: labeler rowwise + street net ----
        for street, labeler_args, extra in [
            ("t1", ["--t1-vs-fl-library", LIBRARY_DIR,
                    "--t1-t2-model", "D:/ofc_data/t2_vs_fl_model_v1/evaluator.bin",
                    "--t2-t3-model", "D:/ofc_data/t3_vs_fl_model_v1/evaluator.bin"],
             {"t2_samples": 1, "t3_samples": 1, "t4_draw_sample": 1}),
            ("t2", ["--t2-vs-fl-library", LIBRARY_DIR,
                    "--t2-t3-model", "D:/ofc_data/t3_vs_fl_model_v1/evaluator.bin"],
             {"t3_samples": 1, "t4_draw_sample": 1}),
        ]:
            draw, cursor = deck[cursor:cursor+3], cursor + 3
            request = {"id": "d", "board": {"top": rows[0], "middle": rows[1], "bottom": rows[2]},
                       "dead": dead, "draw": draw, "opp_count": opp_count, **extra}
            response = run_labeler(labeler_args, [request])[0]
            table = {a["action_key"]: a for a in response["actions"]}
            board = Board(top=list(rows[0]), middle=list(rows[1]), bottom=list(rows[2]))
            candidates, vectors = [], []
            for action in get_turn_actions(list(draw), board):
                key = exact_late.action_key(action)
                if key not in table:
                    continue
                after = exact_late.apply_action(board, action)
                vectors.append(encode_t2_action(
                    (after.top, after.middle, after.bottom),
                    dead + [action.discard], opp_count,
                    table[key]["own_rowwise_block"]))
                candidates.append(action)
            if nets[street].input_dim == 109:
                from ai.tutor.joint_blocks import fetch_joint_blocks
                from ai.tutor.t4_vs_fl import CARD_INDEX as CI
                requests_jb = []
                for position, action in enumerate(candidates):
                    after = exact_late.apply_action(board, action)
                    seen_jb = seen_mask(
                        [c for r in (after.top, after.middle, after.bottom) for c in r]
                        + dead + [action.discard])
                    requests_jb.append({
                        "id": str(position),
                        "board": {"top": list(after.top), "middle": list(after.middle),
                                  "bottom": list(after.bottom)},
                        "pool": [c for c in ALL_CARDS_FULL
                                 if not ((1 << CI[c]) & seen_jb)],
                    })
                blocks = fetch_joint_blocks(requests_jb, Path.cwd())
                vectors = [v[:89] + blocks[str(i)] + v[89:]
                           for i, v in enumerate(vectors)]
            scores = nets[street].predict(vectors)
            best = candidates[int(np.argmax(scores))]
            after = exact_late.apply_action(board, best)
            rows = [list(after.top), list(after.middle), list(after.bottom)]
            dead = dead + [best.discard]
            print(f"{street.upper()} (モデル値 {scores.max():+.2f}):")
            print(show(rows, draw, best.discard))

        # ---- T3: full 109-dim features emitted by the labeler ----
        draw, cursor = deck[cursor:cursor+3], cursor + 3
        response = run_labeler(
            ["--t3-vs-fl-library", LIBRARY_DIR],
            [{"id": "d", "board": {"top": rows[0], "middle": rows[1], "bottom": rows[2]},
              "dead": dead, "draw": draw, "opp_count": opp_count, "draw_sample": 30}],
        )[0]
        import os
        table = {a["action_key"]: a for a in response["actions"]}
        board = Board(top=list(rows[0]), middle=list(rows[1]), bottom=list(rows[2]))
        candidates, vectors = [], []
        for action in get_turn_actions(list(draw), board):
            key = exact_late.action_key(action)
            if key not in table:
                continue
            after = exact_late.apply_action(board, action)
            vectors.append(encode_t3_action(
                (after.top, after.middle, after.bottom),
                dead + [action.discard], opp_count,
                table[key]["own_rowwise_block"], table[key]["own_joint_block"]))
            candidates.append(action)
        scores = nets["t3"].predict(vectors)
        best = candidates[int(np.argmax(scores))]
        after = exact_late.apply_action(board, best)
        rows = [list(after.top), list(after.middle), list(after.bottom)]
        dead = dead + [best.discard]
        print(f"T3 (モデル値 {scores.max():+.2f}):")
        print(show(rows, draw, best.discard))

        # ---- T4: exact best against the FL library ----
        draw, cursor = deck[cursor:cursor+3], cursor + 3
        seen = seen_mask([c for r in rows for c in r] + dead + list(draw))
        compatible = np.where((library.masks & np.uint64(seen)) == np.uint64(0))[0]
        best_value, best_rows, best_discard = -1e18, None, None
        for keep in combinations(range(3), 2):
            discard = next(i for i in range(3) if i not in keep)
            open_slots = [3 - len(rows[0]), 5 - len(rows[1]), 5 - len(rows[2])]
            for r1 in range(3):
                for r2 in range(3):
                    need = [0, 0, 0]
                    need[r1] += 1
                    need[r2] += 1
                    if any(need[i] > open_slots[i] for i in range(3)):
                        continue
                    final = [list(r) for r in rows]
                    final[r1].append(draw[keep[0]])
                    final[r2].append(draw[keep[1]])
                    hero = hero_terminal(final)
                    value = score_against_library(hero, library, compatible, opp_count)
                    if value > best_value:
                        best_value, best_rows, best_discard = value, final, draw[discard]
        rows = best_rows
        hero = hero_terminal(rows)
        print(f"T4 (ライブラリ厳密, EV {best_value:+.2f}, FLサンプル {len(compatible)}):")
        print(show(rows, draw, best_discard))
        state = "バースト!" if hero["busted"] else (
            f"royalty {hero['royalty']}" + (f" / FL進入{hero['entry']}枚!" if hero["entry"] else "")
        )
        print(f"最終: {state}")


if __name__ == "__main__":
    main()
