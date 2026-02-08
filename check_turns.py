import sqlite3, json

conn = sqlite3.connect('data/ofc_logs.db')
cur = conn.cursor()

# Show 3 sample turns
cur.execute("SELECT * FROM turns WHERE action_placements IS NOT NULL ORDER BY rowid LIMIT 3")
cols = [d[0] for d in cur.description]
for row in cur.fetchall():
    print("---")
    for c, v in zip(cols, row):
        if v and len(str(v)) > 120:
            print(f"  {c}: {str(v)[:120]}...")
        else:
            print(f"  {c}: {v}")

# Count valid turns (with placements)
cur.execute("SELECT COUNT(*) FROM turns WHERE action_placements IS NOT NULL")
print(f"\nTotal turns with placements: {cur.fetchone()[0]}")

# Count by turn number
cur.execute("SELECT turn, COUNT(*) FROM turns WHERE action_placements IS NOT NULL GROUP BY turn ORDER BY turn")
for row in cur.fetchall():
    print(f"  Turn {row[0]}: {row[1]} rows")

conn.close()
