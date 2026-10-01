"""Stated-vs-revealed analysis: code model-stated reasons for attribute
mentions, validate coders, compare mention shares to AMCE importance shares.

Coders:
  1. dictionary  - regex lexicons per attribute
  2. judge       - gemma2:2b via Ollama (NOT in audited panel); GPU required,
                   so judge coding runs only with --judge once GPU is free
Validation: 300 stratified reasons exported for author hand-coding
  (--export-validation); per-attribute Cohen's kappa computed on return.

Output: results/tables/reason_codes.csv, stated_vs_revealed.csv

Run:  python -m src.analysis.reasons [--judge] [--export-validation]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.design import load_config

ROOT = Path(__file__).resolve().parents[1].parent

LEXICONS = {
    "rating": r"\brat(?:ed|ing)s?\b|\bscore\b|\b4\.\d\b|stars?\b|highly rated",
    "volume": r"\b(?:number|count|many|few|lots?) of review|\breview count\b|\b(?:12|85|400) review|hundreds of review",
    "recency": r"\brecent\b|\brecency\b|\bup.?to.?date\b|\bcurrent\b|\bfresh\b|\bstale\b|\bmonths? ago\b|\bdays? ago\b|\boutdated\b",
    "response": r"\brespond|\breplies\b|\breply\b|\bresponsive|\bfeedback\b|\bengag(?:es|ed|ement)\b",
    "affiliation": r"\bhospital system\b|\buniversity hospital\b|\bnetwork\b|\bindependent|\bprivate practice\b|\baffiliat",
    "price": r"\bprice\b|\bcost\b|\bbudget\b|\bafford|\bcheap|\bexpensive\b|\bvalue\b|\bfee\b|\$\d+|out.of.pocket",
    "telehealth": r"\btelehealth\b|\btele.?medicine\b|\bvirtual\b|\bvideo visit|\bremote\b",
    "experience": r"\bexperienc|\byears? in practice\b|\bin practice for\b|\bseasoned\b|\bveteran\b|\btenure\b",
    "gender": r"\bfemale\b|\bmale\b|\bwoman\b|\bman\b|\bgender\b",
    "ethnicity": r"\bethnic|\brace\b|\bracial\b|\bcultural background\b|\bdiverse\b|\bdiversity\b",
    "position": r"\bfirst\b|\blisted\b|\btop of\b",
}

JUDGE_PROMPT = """You label doctor-recommendation explanations. For the reason
text below, output ONLY a JSON object with boolean fields indicating whether
the reason refers to each factor:
{{"rating": bool, "volume": bool, "recency": bool, "response": bool,
"affiliation": bool, "price": bool, "telehealth": bool, "experience": bool,
"gender": bool, "ethnicity": bool, "position": bool}}

rating = average patient score; volume = number/count of reviews;
recency = how recent reviews are; response = practice responding to feedback;
affiliation = hospital system vs independent practice; price = fee/cost/value;
telehealth = virtual visits; experience = years in practice;
gender = the doctor's gender; ethnicity = the doctor's ethnicity, race, or
cultural background; position = order in the list.

Reason: "{reason}"
"""


def load_reasons() -> pd.DataFrame:
    cfg = load_config()
    df = pd.read_parquet(ROOT / cfg["paths"]["processed"] / "choices.parquet")
    main = df[(df["arm"] == "main") & (df["chosen"] == 1)]
    return (main[["model", "trial_id", "persona", "reason"]]
            .dropna(subset=["reason"]).drop_duplicates(["model", "trial_id"]))


def dictionary_code(reasons: pd.DataFrame) -> pd.DataFrame:
    out = reasons.copy()
    for attr, pattern in LEXICONS.items():
        out[f"d_{attr}"] = out["reason"].str.contains(
            pattern, case=False, regex=True).astype(int)
    return out


def judge_code(reasons: pd.DataFrame) -> pd.DataFrame:
    """gemma2:2b coding via Ollama. GPU recommended; run when free."""
    import requests
    cfg = load_config()
    judge_id = cfg["reason_judge"]["id"]
    out_rows = []
    ckpt = ROOT / cfg["paths"]["raw"] / "judge_codes.jsonl"
    done = set()
    if ckpt.exists():
        with open(ckpt, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["key"])
                except (json.JSONDecodeError, KeyError):
                    pass
    todo = [(f"{r.model}_{r.trial_id}", r) for r in reasons.itertuples()
            if f"{r.model}_{r.trial_id}" not in done]
    print(f"[judge] {len(todo)} reasons to code ({len(done)} done)")
    for i, (key, r) in enumerate(todo):
        body = {"model": judge_id, "stream": False, "format": "json",
                "messages": [{"role": "user",
                              "content": JUDGE_PROMPT.format(reason=r.reason)}],
                "options": {"temperature": 0.0, "seed": 1}}
        try:
            resp = requests.post("http://localhost:11434/api/chat", json=body,
                                 timeout=60).json()
            obj = json.loads(resp["message"]["content"])
            rec = {"key": key, **{f"j_{a}": int(bool(obj.get(a, False)))
                                  for a in LEXICONS}}
        except Exception:
            rec = {"key": key, **{f"j_{a}": None for a in LEXICONS}}
        with open(ckpt, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        if (i + 1) % 100 == 0:
            print(f"[judge] {i + 1}/{len(todo)}")
    return pd.read_json(ckpt, lines=True)


def stated_vs_revealed(coded: pd.DataFrame, prefix: str) -> pd.DataFrame:
    cfg = load_config()
    shares = (coded.groupby("model")[[f"{prefix}{a}" for a in LEXICONS]]
              .mean().reset_index()
              .melt(id_vars="model", var_name="attribute",
                    value_name="mention_share"))
    shares["attribute"] = shares["attribute"].str.removeprefix(prefix)
    imp_path = ROOT / cfg["paths"]["tables"] / "importance_shares.csv"
    if imp_path.exists():
        imp = pd.read_csv(imp_path)
        mapping = {"rating": "rating", "volume": "volume", "recency": "recency",
                   "response": "response", "affiliation": "affiliation",
                   "price": "price", "telehealth": "telehealth",
                   "experience": "experience", "gender": "gender",
                   "ethnicity": "ethnicity", "position": "position"}
        shares["amce_share"] = shares["attribute"].map(
            lambda a: imp.set_index("attribute")["share"].get(mapping.get(a)))
    return shares


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--export-validation", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    out = ROOT / cfg["paths"]["tables"]
    out.mkdir(parents=True, exist_ok=True)

    reasons = load_reasons()
    coded = dictionary_code(reasons)

    if args.export_validation:
        sample = coded.groupby("model", group_keys=False).apply(
            lambda g: g.sample(min(75, len(g)), random_state=3))
        cols = ["model", "trial_id", "reason"]
        sample[cols].to_csv(out / "validation_sample_for_handcoding.csv",
                            index=False)
        print(f"[reasons] exported {len(sample)} reasons for author coding")

    if args.judge:
        judge = judge_code(reasons)
        coded["key"] = coded["model"].astype(str) + "_" + coded["trial_id"].astype(str)
        coded = coded.merge(judge, on="key", how="left")

    coded.to_csv(out / "reason_codes.csv", index=False)
    svr = stated_vs_revealed(coded, "d_")
    svr.to_csv(out / "stated_vs_revealed.csv", index=False)
    print(f"[reasons] coded {len(coded)} reasons; "
          f"dictionary mention shares written")


if __name__ == "__main__":
    main()
