"""Deduplicate data/raw claude JSONL files, keeping first (trial_id, rep) occurrence."""
import json
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

EXPECTED = {
    "main": 3024, "grounded": 297, "rank_top3": 297, "field_order": 189,
    "temp0": 594, "retest": 540, "letter": 297, "nopersona": 297,
    "card_reversed": 189,
}

for f in sorted(RAW.glob("claude-*__*.jsonl")):
    if "pilot" in f.name:
        continue
    recs = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    before = len(recs)
    seen = set()
    deduped = []
    for r in recs:
        key = (r["trial_id"], r.get("rep", 0))
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    arm = f.stem.split("__")[1]
    exp = EXPECTED.get(arm, "?")
    status = "OK" if len(deduped) == exp else "MISMATCH"
    f.write_text("\n".join(json.dumps(r) for r in deduped) + "\n", encoding="utf-8")
    print(f"{f.name}: {before} -> {len(deduped)} (expected {exp}) {status}")
