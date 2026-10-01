"""Decompose position bias from letter-token bias in the letter/logprob arm.

The letter arm asks the model for a single capital letter A-E. Because each
doctor's display slot (1-5) and its letter label are independently randomized
in the design, regressing the model's first-token letter probability on slot
dummies and letter dummies separates a genuine list-position effect from any
intrinsic preference for particular letter tokens.

Pipeline:
  1. Read data/raw/*__letter.jsonl, keep ok==true rows with non-null logprobs.
  2. For each call, pull the first generated token's top-k alternatives, map
     tokens to letters A-E, and renormalize exp(logprob) over the matched
     letters into a probability per letter (coverage = matched probability mass).
  3. Merge the design (letter -> slot, attributes) and fit
     prob ~ C(slot) + C(letter), trial-clustered SEs.
  4. Write results/tables/letter_decomposition.csv (coefficients + sequential
     partial R^2 for the slot block vs the letter block).

Handles both OpenAI-compatible and (defensively) Ollama logprob structures;
unrecognized structures are logged and skipped. If no usable logprob rows are
found, writes a single note row and exits 0.

Run:  python -m src.analysis.letterarm
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from src.design import load_config

ROOT = Path(__file__).resolve().parents[1].parent
LETTERS = ["A", "B", "C", "D", "E"]


# --------------------------------------------------------------------------
# Logprob structure handling
# --------------------------------------------------------------------------

def token_to_letter(token: str) -> str | None:
    """Strip whitespace/punctuation; return the letter if a single A-E remains."""
    if not isinstance(token, str):
        return None
    s = token.strip().strip(".:)-(]\"'").strip()
    if len(s) == 1 and s.upper() in LETTERS:
        return s.upper()
    return None


def first_token_alternatives(logprobs) -> list[dict] | None:
    """Return the first generated token's alternatives as [{token, logprob}, ...].

    Recognizes:
      - OpenAI-compat: {"content": [{"top_logprobs": [{"token", "logprob"}...]}]}
      - Ollama (defensive guesses): a top-level list of per-position dicts each
        carrying "top_logprobs", or a dict with "tokens"/"top_logprobs".
    Returns None if the structure is not recognized.
    """
    if logprobs is None:
        return None

    # OpenAI-compatible shape
    if isinstance(logprobs, dict) and isinstance(logprobs.get("content"), list):
        content = logprobs["content"]
        if not content:
            return None
        top = content[0].get("top_logprobs")
        if isinstance(top, list) and top and "logprob" in top[0]:
            return [{"token": t.get("token"), "logprob": t.get("logprob")}
                    for t in top]
        # fall back to the single chosen token if no alternatives are given
        if "token" in content[0] and "logprob" in content[0]:
            return [{"token": content[0]["token"],
                     "logprob": content[0]["logprob"]}]
        return None

    # Defensive Ollama-style guesses (structure not confirmed upstream)
    if isinstance(logprobs, list) and logprobs:
        first = logprobs[0]
        if isinstance(first, dict):
            top = first.get("top_logprobs") or first.get("top")
            if isinstance(top, list) and top and isinstance(top[0], dict) \
                    and "logprob" in top[0]:
                return [{"token": t.get("token") or t.get("tok"),
                         "logprob": t.get("logprob")} for t in top]
    if isinstance(logprobs, dict) and isinstance(logprobs.get("top_logprobs"),
                                                 list):
        top = logprobs["top_logprobs"]
        if top and isinstance(top[0], dict) and "logprob" in top[0]:
            return [{"token": t.get("token"), "logprob": t.get("logprob")}
                    for t in top]

    return None


def letter_vector(alts: list[dict]) -> tuple[dict[str, float], float] | None:
    """Map alternatives to letters and renormalize over matched mass.

    Returns ({letter: prob}, coverage) where coverage is the matched probability
    mass before renormalization, or None if no token mapped to a letter.
    """
    mass = {ltr: 0.0 for ltr in LETTERS}
    for a in alts:
        lp = a.get("logprob")
        ltr = token_to_letter(a.get("token"))
        if ltr is None or lp is None:
            continue
        mass[ltr] += math.exp(float(lp))
    coverage = sum(mass.values())
    if coverage <= 0:
        return None
    probs = {ltr: mass[ltr] / coverage for ltr in LETTERS}
    return probs, coverage


# --------------------------------------------------------------------------
# Data assembly
# --------------------------------------------------------------------------

def build_long(cfg: dict) -> pd.DataFrame:
    raw_dir = ROOT / cfg["paths"]["raw"]
    doctors = pd.read_parquet(ROOT / cfg["paths"]["design"] / "doctors.parquet")

    rows = []
    if not raw_dir.exists():
        return pd.DataFrame(rows)

    for path in sorted(raw_dir.glob("*__letter.jsonl")):
        if path.stem.endswith("_pilot"):
            continue
        recognized_any = False
        unrecognized = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not rec.get("ok") or rec.get("logprobs") is None:
                    continue
                alts = first_token_alternatives(rec["logprobs"])
                if alts is None:
                    unrecognized += 1
                    continue
                lv = letter_vector(alts)
                if lv is None:
                    continue
                recognized_any = True
                probs, coverage = lv
                tid = rec["trial_id"]
                doc = doctors[doctors["trial_id"] == tid]
                slot_of = dict(zip(doc["letter"], doc["slot"]))
                for ltr in LETTERS:
                    if ltr not in slot_of:
                        continue
                    rows.append(dict(model=rec["model"], trial_id=tid,
                                     letter=ltr, slot=int(slot_of[ltr]),
                                     prob=probs[ltr], coverage=coverage))
        if not recognized_any:
            extra = (f" (unrecognized logprob structure in {unrecognized} records)"
                     if unrecognized else "")
            print(f"[letterarm] {path.name}: no usable logprob rows{extra}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Estimation
# --------------------------------------------------------------------------

def _r2(df: pd.DataFrame, formula: str) -> float:
    return smf.ols(formula, data=df).fit().rsquared


def decompose(long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (coefficient table, sequential partial-R^2 table)."""
    res = smf.ols("prob ~ C(slot) + C(letter)", data=long).fit(
        cov_type="cluster", cov_kwds={"groups": long["trial_id"]})
    coef_rows = []
    for name in res.params.index:
        if name == "Intercept":
            continue
        coef_rows.append(dict(term=name,
                              coef_pp=round(100 * res.params[name], 3),
                              se_pp=round(100 * res.bse[name], 3),
                              p=round(float(res.pvalues[name]), 4)))
    coef = pd.DataFrame(coef_rows)

    # Sequential partial R^2: how much each block adds on top of the other.
    r2_slot = _r2(long, "prob ~ C(slot)")
    r2_letter = _r2(long, "prob ~ C(letter)")
    r2_full = _r2(long, "prob ~ C(slot) + C(letter)")
    partial = pd.DataFrame([
        dict(block="slot | letter", added_r2=round(r2_full - r2_letter, 4),
             block_only_r2=round(r2_slot, 4)),
        dict(block="letter | slot", added_r2=round(r2_full - r2_slot, 4),
             block_only_r2=round(r2_letter, 4)),
        dict(block="full", added_r2=round(r2_full, 4), block_only_r2=round(r2_full, 4)),
    ])
    return coef, partial


def main() -> None:
    cfg = load_config()
    out = ROOT / cfg["paths"]["tables"]
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "letter_decomposition.csv"

    long = build_long(cfg)
    if long.empty:
        pd.DataFrame([dict(note="no logprob data captured; letter arm analyzed "
                                "via sampled choices only")]).to_csv(dest,
                                                                      index=False)
        print(f"[letterarm] no usable logprob rows; wrote note to {dest}")
        return

    coef, partial = decompose(long)
    # Stack the two blocks into one CSV with a 'section' marker column.
    coef.insert(0, "section", "coefficient")
    partial = partial.rename(columns={"block": "term"})
    partial.insert(0, "section", "partial_r2")
    combined = pd.concat([coef, partial], ignore_index=True)
    combined.to_csv(dest, index=False)

    n_calls = long.groupby(["model", "trial_id"]).ngroups
    print(f"[letterarm] {len(long)} letter-rows from {n_calls} calls; "
          f"mean coverage={long['coverage'].mean():.3f}")
    print(f"[letterarm] wrote {dest}")


if __name__ == "__main__":
    main()
