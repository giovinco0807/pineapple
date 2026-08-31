import json
import subprocess
from pathlib import Path

exe_path = str(Path(__file__).resolve().parent.parent.parent / "ai" / "rust_solver" / "target" / "release" / "t3_exact.exe")

req = {
    "board_top": ["Kh", "X2"],
    "board_mid": ["7c", "2d", "8s", "3h"],
    "board_bot": ["Qc", "8h", "Th", "Ts", "8c"],
    "discards": ["9h", "9s", "Qd"],
    "dealt": [], # This is actually a T4 state if dealt is empty, but we want to evaluate T4 draws from this T3 state
    "bust_penalty": -4.0,
    "fl_ev": {"14": 24.92, "15": 31.43, "16": 39.77, "17": 47.74}
}

# Wait, if we want to see why action 7 gives -4.0, we can run t3_exact on the T3 state again, 
# and maybe add a print in the rust code, or just run a specific T4 board in python to see if rust says busted.

# Let's test a specific T4 draw for this action
# Top: Kh X2
# Mid: 7c 2d 8s 3h
# Bot: Qc 8h Th Ts 8c
# Draw: Ac, 4d, 5s

# If we put Ac in Mid, 4d in Top, 5s discard.
# Top: Kh X2 4d
# Mid: 7c 2d 8s 3h Ac
# Bot: Qc 8h Th Ts 8c

# Let's call a python script to check this.
