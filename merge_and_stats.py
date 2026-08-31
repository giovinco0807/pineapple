"""
merge_and_stats.py
  1. Downloads/teacher_results_vm*.tar.gz を展開
  2. mc_teacher/ に全shardをコピー
  3. mc_s50_merged.jsonl を生成
  4. 統計を表示してログに保存
"""

import os, glob, json, tarfile, shutil
import numpy as np

DOWNLOADS = r"C:\Users\Owner\Downloads"
DEST      = r"D:\ofc_data\mc_teacher"
SIMS      = 50

os.makedirs(DEST, exist_ok=True)

# ── 1. Extract all tar.gz from Downloads ──────────────────────────────────
tar_files = sorted(glob.glob(os.path.join(DOWNLOADS, "teacher_results_vm*.tar.gz")))
print(f"Found {len(tar_files)} tar archives in Downloads:")
for tf in tar_files:
    sz = os.path.getsize(tf) / 1024 / 1024
    print(f"  {os.path.basename(tf)}  ({sz:.1f} MB)")

print()
for tf in tar_files:
    vm_id = os.path.basename(tf).replace("teacher_results_vm", "").replace(".tar.gz", "")
    extract_dir = os.path.join(DEST, f"_vm{vm_id}_extract")
    os.makedirs(extract_dir, exist_ok=True)
    print(f"Extracting {os.path.basename(tf)} ...", end=" ", flush=True)
    with tarfile.open(tf, "r:gz") as t:
        t.extractall(extract_dir)
    # Move jsonl files to DEST with VM prefix in case of name clash
    moved = 0
    for root, dirs, files in os.walk(extract_dir):
        for fname in files:
            if fname.endswith(".jsonl"):
                src = os.path.join(root, fname)
                # Keep original filename (shard IDs are globally unique by design)
                dst = os.path.join(DEST, fname)
                shutil.copy2(src, dst)
                moved += 1
    shutil.rmtree(extract_dir)
    print(f"{moved} jsonl files")

# ── 2. List all shards ────────────────────────────────────────────────────
shards = sorted(glob.glob(os.path.join(DEST, f"mc_s{SIMS}_shard*.jsonl")))
print(f"\nTotal shards found: {len(shards)}")

# ── 3. Merge ─────────────────────────────────────────────────────────────
merged_path = os.path.join(DEST, f"mc_s{SIMS}_merged.jsonl")
total_lines = 0
print(f"Merging → {merged_path} ...", end=" ", flush=True)
with open(merged_path, "w", encoding="utf-8") as fout:
    for shard in shards:
        with open(shard, "r", encoding="utf-8-sig") as fin:
            for line in fin:
                line = line.strip()
                if line:
                    fout.write(line + "\n")
                    total_lines += 1
print(f"{total_lines} records")

# ── 4. Stats ─────────────────────────────────────────────────────────────
summary_records = []
turn_records    = []
parse_errors    = 0

with open(merged_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            parse_errors += 1
            continue
        turn = d.get("turn", -99)
        if turn == -1:
            summary_records.append(d)
        elif turn >= 0:
            turn_records.append(d)

n = len(summary_records)
if n == 0:
    print("No summary records (turn=-1) found. Check data format.")
else:
    busts   = sum(1 for r in summary_records if r.get("busted", False))
    fls     = sum(1 for r in summary_records if r.get("fl_entry", False))
    scores  = np.array([r["score"] for r in summary_records])
    se      = scores.std() / np.sqrt(n) if n > 1 else 0

    lines_out = []
    lines_out.append("")
    lines_out.append("=" * 55)
    lines_out.append("  MC Teacher Data  ─  Statistics")
    lines_out.append("=" * 55)
    lines_out.append(f"  Hands (summary records):  {n:,}  / 10,000 target")
    lines_out.append(f"  Total records:            {total_lines:,}")
    lines_out.append(f"  Parse errors:             {parse_errors}")
    lines_out.append(f"  Shards collected:         {len(shards)} / 80")
    lines_out.append("")
    lines_out.append(f"  Score   mean  : {scores.mean():+.3f}")
    lines_out.append(f"  Score   std   : {scores.std():.3f}")
    lines_out.append(f"  Score   95%CI : [{scores.mean()-1.96*se:+.3f}, {scores.mean()+1.96*se:+.3f}]")
    lines_out.append(f"  Score   min   : {scores.min():+.1f}")
    lines_out.append(f"  Score   max   : {scores.max():+.1f}")
    lines_out.append("")
    lines_out.append(f"  Bust rate     : {busts/n*100:.2f}%  ({busts}/{n})")
    lines_out.append(f"  FL entry rate : {fls/n*100:.2f}%  ({fls}/{n})")
    lines_out.append("")
    # Per-turn stats
    lines_out.append("  Turn records breakdown:")
    for t in range(6):
        tr = [r for r in turn_records if r.get("turn") == t]
        if tr:
            modes = set(r.get("eval_mode", "?") for r in tr)
            lines_out.append(f"    Turn {t}: {len(tr):,} records  modes={modes}")
    lines_out.append("=" * 55)

    report = "\n".join(lines_out)
    print(report)

    # Save stats
    stats_path = os.path.join(DEST, "stats.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nStats saved → {stats_path}")
    print(f"Merged data → {merged_path}")
