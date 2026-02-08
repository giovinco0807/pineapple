import sqlite3, json, sys

conn = sqlite3.connect('data/ofc_logs.db')
cur = conn.cursor()

out = []

cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in cur.fetchall()]
for t in tables:
    cur.execute(f"PRAGMA table_info([{t}])")
    cols = [(r[1], r[2]) for r in cur.fetchall()]
    cur.execute(f"SELECT COUNT(*) FROM [{t}]")
    cnt = cur.fetchone()[0]
    out.append(f"\n{t} ({cnt} rows):")
    for name, typ in cols:
        out.append(f"  {name} ({typ})")

out.append("\n\n--- Sample hand ---")
cur.execute("SELECT * FROM hands ORDER BY rowid DESC LIMIT 1")
cols = [d[0] for d in cur.description]
row = cur.fetchone()
if row:
    for c, v in zip(cols, row):
        val_str = str(v)[:200] if v else "NULL"
        out.append(f"  {c}: {val_str}")

conn.close()

with open('db_schema.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
print("Written to db_schema.txt")
