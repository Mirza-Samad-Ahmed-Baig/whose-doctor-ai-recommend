import json
from pathlib import Path
import os

raw = Path(__file__).resolve().parents[1] / "data" / "raw"
total_n = total_fail = 0

files = sorted(f for f in raw.iterdir()
               if f.is_file() and f.name.startswith("claude-") and "pilot" not in f.name)

for f in files:
    recs = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    n = len(recs)
    fails = sum(1 for r in recs if not r.get("ok", False))
    pct = 100 * fails / n if n else 0
    flag = " *** HIGH" if pct > 2 else ""
    total_n += n
    total_fail += fails
    print(f"{f.name}: {fails}/{n} ({pct:.1f}%){flag}")

if total_n:
    print(f"\nOVERALL: {total_fail}/{total_n} ({100*total_fail/total_n:.1f}%)")
else:
    print("No files found.")
