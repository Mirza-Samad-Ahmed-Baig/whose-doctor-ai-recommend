"""Build the long-format analysis dataset from raw JSONL checkpoints.

Output: data/processed/choices.parquet  (one row per call x card)
        results/tables/parse_failures.csv, balance.csv

Run:  python -m src.analysis.prep
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.design import load_config, ATTRS

ROOT = Path(__file__).resolve().parents[1].parent


def read_raw(cfg: dict) -> pd.DataFrame:
    rows = []
    raw_dir = ROOT / cfg["paths"]["raw"]
    for path in sorted(raw_dir.glob("*.jsonl")):
        pilot = path.stem.endswith("_pilot")
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rec["pilot"] = pilot
                rec.pop("logprobs", None)  # analyzed separately in letter arm
                rows.append(rec)
    if not rows:
        raise SystemExit("No raw data found under data/raw/")
    df = pd.DataFrame(rows)
    # normalise trial_id to int (some providers write it as string)
    df["trial_id"] = pd.to_numeric(df["trial_id"], errors="coerce").astype("Int64")
    if "rep" not in df.columns:
        df["rep"] = 0
    # coerce reason to string (some providers return a dict or nested object)
    if "reason" in df.columns:
        df["reason"] = df["reason"].apply(lambda x: str(x) if not isinstance(x, (str, type(None))) else x)
    # de-duplicate on key (keep last attempt record)
    df = df.drop_duplicates(subset=["model", "arm", "trial_id", "rep", "pilot"],
                            keep="last")
    return df


def build_long(cfg: dict, calls: pd.DataFrame) -> pd.DataFrame:
    design_dir = ROOT / cfg["paths"]["design"]
    trials = pd.read_parquet(design_dir / "trials.parquet")
    doctors = pd.read_parquet(design_dir / "doctors.parquet")

    ok = calls[(calls["ok"]) & (calls["arm"] != "manip") & (~calls["pilot"])]
    # drop design columns already in raw records to avoid _x/_y duplicates on merge
    ok = ok.drop(columns=[c for c in ("persona", "template_id") if c in ok.columns])
    long = ok.merge(doctors, on="trial_id", how="left")
    long["chosen"] = (long["name"] == long["choice"]).astype(int)
    long = long.merge(trials[["trial_id", "persona", "template_id"]],
                      on="trial_id", how="left")

    # analysis coding (telehealth is recoded in place: present/absent -> 1/0)
    long["recent"] = (long["recency"] == "3 days ago").astype(int)
    long["responds"] = (long["feedback_response"] == "present").astype(int)
    long["network"] = (long["affiliation"] == "network").astype(int)
    long["telehealth"] = (long["telehealth"] == "present").astype(int)
    long["female"] = (long["gender"] == "female").astype(int)
    for eth in ("black", "hispanic", "east_asian", "south_asian"):  # ref: white
        long[f"eth_{eth}"] = (long["ethnicity"] == eth).astype(int)
    return long


def main() -> None:
    cfg = load_config()
    calls = read_raw(cfg)

    # parse-failure table (per model x arm, excluding pilot)
    live = calls[~calls["pilot"]]
    fails = (live[live["arm"] != "manip"]
             .groupby(["model", "arm"])
             .agg(n_calls=("ok", "size"), n_fail=("ok", lambda s: int((~s).sum())))
             .reset_index())
    fails["fail_rate"] = (fails["n_fail"] / fails["n_calls"]).round(4)

    long = build_long(cfg, calls)

    out_t = ROOT / cfg["paths"]["tables"]
    out_p = ROOT / cfg["paths"]["processed"]
    out_t.mkdir(parents=True, exist_ok=True)
    out_p.mkdir(parents=True, exist_ok=True)

    fails.to_csv(out_t / "parse_failures.csv", index=False)
    long.to_parquet(out_p / "choices.parquet", index=False)

    # balance table on analyzed sample (main arm)
    main_arm = long[long["arm"] == "main"]
    bal = []
    for attr in ATTRS:
        shares = main_arm[attr].value_counts(normalize=True).sort_index()
        for lvl, sh in shares.items():
            bal.append(dict(attribute=attr, level=lvl, share=round(sh, 4)))
    pd.DataFrame(bal).to_csv(out_t / "balance.csv", index=False)

    print(f"calls: {len(calls)} | analyzed rows: {len(long)} "
          f"({long['trial_id'].nunique()} trials x models/arms)")
    print(fails.to_string(index=False))


if __name__ == "__main__":
    main()
