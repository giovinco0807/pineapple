"""
OFC Pineapple - Collect Human Play Data

Exports human play logs from the SQLite database to JSONL format
compatible with the BC training pipeline.

Usage:
    python ai/training/collect_human_data.py
    python ai/training/collect_human_data.py --db data/ofc_logs.db --output data/ryo_hands.jsonl
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.db.writer import LogWriter


def main():
    parser = argparse.ArgumentParser(description="Export human play logs for BC training")
    parser.add_argument("--db", default="data/ofc_logs.db", help="SQLite database path")
    parser.add_argument("--output", default="data/ryo_hands.jsonl", help="Output JSONL path")
    args = parser.parse_args()

    db = LogWriter(args.db)
    output = db.export_to_jsonl(args.output)
    print(f"Done! Exported to {output}")


if __name__ == "__main__":
    main()
