"""Convert session-model pilot responses into runner-schema JSONL + summary.

Input:  responses.json  — list of {model, trial_id, raw, attempts}
Output: data/raw_session_pilot/<model>__main_pilot.jsonl  (runner schema)
        stdout summary: parse rate + P(chosen | level shown) per attribute.

Responses were collected via Claude Code session subagents (NOT the API):
no temperature/top_p/seed control, agent harness system prompt present.
Parsing reuses the project's own parse_response for fidelity.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(r"C:\Research paepr 7\dist\physician_audit_frontier_runner")
sys.path.insert(0, str(ROOT))

from src.runner import load_config, load_design, parse_response

cfg = load_config()
trials, doctors = load_design(cfg)

responses = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
outdir = ROOT / "data" / "raw_session_pilot"
outdir.mkdir(parents=True, exist_ok=True)

by_model: dict[str, list] = {}
for r in responses:
    by_model.setdefault(r["model"], []).append(r)

ATTRS = ["gender", "ethnicity", "rating", "price", "volume", "recency",
         "affiliation", "experience", "telehealth", "feedback_response", "slot"]

for model, recs in sorted(by_model.items()):
    rows = []
    for r in sorted(recs, key=lambda x: x["trial_id"]):
        tid = int(r["trial_id"])
        doc = doctors[doctors["trial_id"] == tid]
        names = doc.sort_values("slot")["name"].tolist()
        raw = r.get("raw") or ""
        parsed = parse_response(raw, "choice_first", names, {})
        chosen_slot = None
        if parsed["choice"] is not None:
            chosen_slot = int(doc.loc[doc["name"] == parsed["choice"], "slot"].iloc[0])
        rows.append(dict(
            trial_id=tid, rep=0, arm="main_pilot_session", model=model,
            seed=None, raw=raw, ok=parsed["ok"], choice=parsed["choice"],
            chosen_slot=chosen_slot, reason=parsed["reason"], ranking=None,
            attempts=r.get("attempts", 1),
            collection="claude_code_session_subagent (no temp/seed control)",
            ts=datetime.now(timezone.utc).isoformat()))

    safe = model.replace("/", "_").replace(":", "_")
    p = outdir / f"{safe}__main_pilot.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    n = len(rows)
    fails = sum(1 for x in rows if not x["ok"])
    print(f"\n===== {model}: {n} trials, parse failures {fails}/{n} =====")

    ok_tids = {x["trial_id"]: x["chosen_slot"] for x in rows if x["ok"]}
    sub = doctors[doctors["trial_id"].isin(ok_tids)].copy()
    sub["chosen"] = [
        int(s == ok_tids[t]) for t, s in zip(sub["trial_id"], sub["slot"])]
    base = 1.0 / cfg["choice_set"]["n_alternatives"]
    for attr in ATTRS:
        if attr not in sub.columns:
            continue
        parts = []
        for lvl, grp in sub.groupby(attr, observed=True):
            parts.append(f"{lvl}: {grp['chosen'].mean():.3f} (n={len(grp)})")
        print(f"  P(chosen) by {attr} [null={base:.2f}] -> " + "; ".join(parts))

print("\nwrote JSONL files to", outdir)
