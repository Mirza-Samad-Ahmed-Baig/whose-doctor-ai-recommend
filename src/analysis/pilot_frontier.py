"""Exploratory frontier-model session pilot: descriptives for the paper.

Input:  dist/physician_audit_frontier_runner/data/raw_session_pilot/*.jsonl
        (200 real records: 50 main-arm pilot trials x 4 Claude-family models,
        collected interactively in a Claude Code session -- see the
        README_DATA_PROVENANCE.txt in that folder). These records are
        EXPLORATORY ONLY and are never merged into data/raw/ or into any
        confirmatory estimate; this module reads them from their quarantined
        location and produces a clearly-labelled descriptive table.

Output: results/tables/pilot_frontier.csv
        paper/tables/tab_pilot_frontier.tex

Run:  python -m src.analysis.pilot_frontier
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.analysis.tables import _esc, _table

ROOT = Path(__file__).resolve().parents[1].parent
# Canonical quarantine location; fall back to the dist extract it was first
# delivered in. Both are OUTSIDE data/raw/ by design.
PILOT_DIR = ROOT / "data" / "raw_session_pilot"
if not PILOT_DIR.exists():
    PILOT_DIR = (ROOT / "dist" / "physician_audit_frontier_runner" / "data"
                 / "raw_session_pilot")

# Column order in the table (session alias -> display header).
MODELS = [
    ("claude-haiku-4-5", "Haiku 4.5"),
    ("claude-sonnet-5", "Sonnet 5"),
    ("claude-opus-4-8", "Opus 4.8"),
    ("claude-fable-5", "Fable 5"),
]

# (attribute column, level value, row label) -- null P(chosen) is 0.20.
ROWS = [
    ("rating", "3.9", "Rating 3.9"),
    ("rating", "4.3", "Rating 4.3"),
    ("rating", "4.7", "Rating 4.7"),
    ("price", "90", "Fee \\$90"),
    ("price", "140", "Fee \\$140"),
    ("price", "190", "Fee \\$190"),
    ("volume", "12", "12 reviews"),
    ("volume", "85", "85 reviews"),
    ("volume", "400", "400 reviews"),
    ("recency", "3 days ago", "Recent review (3 days)"),
    ("recency", "11 months ago", "Stale review (11 mo.)"),
    ("gender", "female", "Female name"),
    ("gender", "male", "Male name"),
    ("ethnicity", "white", "White name"),
    ("ethnicity", "black", "Black name"),
    ("ethnicity", "hispanic", "Hispanic name"),
    ("ethnicity", "east_asian", "East Asian name"),
    ("ethnicity", "south_asian", "South Asian name"),
]

GROUP_BREAKS = {"Fee \\$90", "12 reviews", "Recent review (3 days)",
                "Female name", "White name"}


def load_pilot_long() -> pd.DataFrame:
    doctors = pd.read_parquet(ROOT / "data" / "design" / "doctors.parquet")
    frames = []
    for model_id, _ in MODELS:
        path = PILOT_DIR / f"session_{model_id}__main_pilot.jsonl"
        recs = [json.loads(line) for line in
                path.read_text(encoding="utf-8").splitlines() if line.strip()]
        calls = pd.DataFrame(recs)
        if not bool(calls["ok"].all()):
            raise SystemExit(f"unexpected parse failure in {path.name}")
        long = calls.merge(doctors, on="trial_id", how="left")
        long["chosen"] = (long["name"] == long["choice"]).astype(int)
        long["model_id"] = model_id
        frames.append(long)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    long = load_pilot_long()
    n_trials = long.groupby("model_id")["trial_id"].nunique()

    rows = []
    for attr, level, label in ROWS:
        row = {"attribute": attr, "level": level, "label": label}
        for model_id, _ in MODELS:
            sub = long[(long["model_id"] == model_id)
                       & (long[attr].astype(str) == level)]
            row[model_id] = sub["chosen"].mean()
            row[f"{model_id}_n"] = len(sub)
        rows.append(row)
    out = pd.DataFrame(rows)

    tdir = ROOT / "results" / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    out.to_csv(tdir / "pilot_frontier.csv", index=False)

    lines = []
    for _, r in out.iterrows():
        if r["label"] in GROUP_BREAKS:
            lines.append("\\addlinespace")
        cells = " & ".join(f"{r[m]:.2f}" for m, _ in MODELS)
        lines.append(f"{r['label']} & {cells} \\\\")
    body = ("\\begin{tabular}{l" + "r" * len(MODELS) + "}\n\\toprule\n"
            " & " + " & ".join(_esc(h) for _, h in MODELS) + " \\\\\n"
            "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n"
            "\\end{tabular}")
    caption = (
        "Exploratory session pilot of four Claude-family models: share of "
        "cards chosen by attribute level (null $=0.20$). Each model saw the "
        "first 50 main-arm choice sets of the frozen design (250 cards; "
        "0 parse failures). Responses were collected interactively inside an "
        "agentic coding harness with default decoding parameters, not via "
        "the Messages API under the audit protocol; estimates are "
        "descriptive and exploratory, are quarantined from the confirmatory "
        "pipeline, and enter no confirmatory test.")
    tex = _table(body, caption, "pilot_frontier")
    dest = ROOT / "paper" / "tables" / "tab_pilot_frontier.tex"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(tex, encoding="utf-8")
    print(f"[pilot_frontier] trials per model: {dict(n_trials)}")
    print(f"[pilot_frontier] wrote {dest} and {tdir / 'pilot_frontier.csv'}")


if __name__ == "__main__":
    main()
