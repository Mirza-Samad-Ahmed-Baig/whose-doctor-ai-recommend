"""Primary analysis: AMCEs via linear probability models, trial-clustered SEs.

Estimates per model and pooled (model fixed effects), confirmatory H1-H12
with Holm correction, attribute importance shares, and TOST equivalence
helpers.

Output: results/tables/amce_{scope}.csv, confirmatory_tests.csv,
        importance_shares.csv

Run:  python -m src.analysis.amce [--arm main]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from src.design import load_config

ROOT = Path(__file__).resolve().parents[1].parent

BASE_FORMULA = ("chosen ~ C(rating) + C(volume) + recent + responds + network "
                "+ C(price) + telehealth + C(experience) + female"
                " + eth_black + eth_hispanic + eth_east_asian + eth_south_asian"
                " + C(slot)")
ETH_TERMS = ["eth_black", "eth_hispanic", "eth_east_asian", "eth_south_asian"]
INTERACT_FORMULA = (BASE_FORMULA
                    + " + C(rating):C(volume)"
                    + " + C(persona):C(price) + C(persona):telehealth"
                    + " + C(persona):C(experience)"
                    + "".join(f" + female:{e}" for e in ETH_TERMS))

CONFIRMATORY = {  # H -> wald constraint pattern (regex on param names)
    "H1_rating": r"C\(rating\)\[T\.",
    "H2_volume": r"C\(volume\)\[T\.",
    "H3_recency": r"^recent$",
    "H4_response": r"^responds$",
    "H5_network": r"^network$",
    "H6_price": r"C\(price\)\[T\.",
    "H7_telehealth": r"^telehealth$",
    "H8_experience": r"C\(experience\)\[T\.",
    "H9_gender": r"^female$",
    "H10_ethnicity": r"^eth_",
    "H11_persona_x_price": r"C\(persona\).*:C\(price\)",
    "H12_rating_x_volume": r"C\(rating\).*:C\(volume\)",
    "H13_intersectional": r"female:eth_",  # gender x ethnicity non-additivity
}
# H11-H13 are tested on the interacted model; H1-H10 on the base model.
FIRST_INTERACTION_H = 11


def load_choices(arm: str) -> pd.DataFrame:
    cfg = load_config()
    df = pd.read_parquet(ROOT / cfg["paths"]["processed"] / "choices.parquet")
    return df[df["arm"] == arm].copy()


def fit_lpm(df: pd.DataFrame, formula: str):
    return smf.ols(formula, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["trial_id"]})


def amce_table(res, scope: str) -> pd.DataFrame:
    rows = []
    for name, coef in res.params.items():
        if name == "Intercept" or ":" in name:
            continue
        rows.append(dict(scope=scope, term=name,
                         amce_pp=round(100 * coef, 2),
                         se_pp=round(100 * res.bse[name], 2),
                         ci_lo_pp=round(100 * (coef - 1.96 * res.bse[name]), 2),
                         ci_hi_pp=round(100 * (coef + 1.96 * res.bse[name]), 2),
                         p=res.pvalues[name]))
    return pd.DataFrame(rows)


def joint_test(res, pattern: str) -> tuple[float, float, int]:
    import re
    names = [n for n in res.params.index if re.search(pattern, n)]
    if not names:
        return np.nan, np.nan, 0
    R = np.zeros((len(names), len(res.params)))
    for i, n in enumerate(names):
        R[i, list(res.params.index).index(n)] = 1
    w = res.wald_test(R, scalar=True)
    return float(w.statistic), float(w.pvalue), len(names)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj


def importance_shares(res) -> pd.DataFrame:
    groups = {
        "rating": [n for n in res.params.index if n.startswith("C(rating)")],
        "volume": [n for n in res.params.index if n.startswith("C(volume)")],
        "price": [n for n in res.params.index if n.startswith("C(price)")],
        "experience": [n for n in res.params.index
                       if n.startswith("C(experience)")],
        "recency": ["recent"], "response": ["responds"],
        "affiliation": ["network"], "telehealth": ["telehealth"],
        "gender": ["female"],
        "ethnicity": [n for n in res.params.index if n.startswith("eth_")],
        "position": [n for n in res.params.index if n.startswith("C(slot)")],
    }
    ranges = {}
    for g, names in groups.items():
        vals = [0.0] + [res.params[n] for n in names if n in res.params.index]
        ranges[g] = max(vals) - min(vals)
    total = sum(ranges.values())
    return pd.DataFrame([dict(attribute=g, range_pp=round(100 * r, 2),
                              share=round(r / total, 3))
                         for g, r in ranges.items()])


def tost(amce: float, se: float, sesoi: float = 0.015) -> float:
    """p-value for equivalence within +/- sesoi (probability scale)."""
    p1 = 1 - stats.norm.cdf((amce + sesoi) / se)
    p2 = stats.norm.cdf((amce - sesoi) / se)
    return max(p1, p2)


CELL_ORDER = [f"{g}_{e}" for g in ("male", "female")
              for e in ("white", "black", "hispanic", "east_asian",
                        "south_asian")]  # male_white is the reference cell


def intersectional_table(df: pd.DataFrame) -> pd.DataFrame:
    """AMCE of each gender x ethnicity cell vs the male-White reference.

    A single 10-level categorical replaces the additive female/eth_* block, so
    each coefficient is the intersectional effect (main effects + interaction)
    with a directly estimated cluster-robust SE. TOST equivalence at +/-1.5 pp
    is reported per cell alongside the directional p-value.
    """
    d = df.copy()
    d["demo_cell"] = pd.Categorical(
        d["gender"].astype(str) + "_" + d["ethnicity"].astype(str),
        categories=CELL_ORDER, ordered=False)
    # additive demographic terms are dropped; the cell dummies subsume them
    reduced = (BASE_FORMULA
               .replace(" + female", "")
               .replace(" + eth_black + eth_hispanic + eth_east_asian"
                        " + eth_south_asian", "")
               + " + C(demo_cell)")
    formula = reduced + (" + C(model)" if d["model"].nunique() > 1 else "")
    res = fit_lpm(d, formula)
    rows = [dict(cell="male_white", amce_pp=0.0, se_pp=0.0, ci_lo_pp=0.0,
                 ci_hi_pp=0.0, p=np.nan, tost_p=np.nan)]
    for cell in CELL_ORDER[1:]:
        name = f"C(demo_cell)[T.{cell}]"
        if name not in res.params.index:
            continue
        coef, se = res.params[name], res.bse[name]
        rows.append(dict(cell=cell, amce_pp=round(100 * coef, 2),
                         se_pp=round(100 * se, 2),
                         ci_lo_pp=round(100 * (coef - 1.96 * se), 2),
                         ci_hi_pp=round(100 * (coef + 1.96 * se), 2),
                         p=float(res.pvalues[name]),
                         tost_p=round(float(tost(coef, se)), 4)))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="main")
    args = ap.parse_args()

    cfg = load_config()
    out = ROOT / cfg["paths"]["tables"]
    out.mkdir(parents=True, exist_ok=True)

    df = load_choices(args.arm)
    tables = []

    for model, grp in df.groupby("model"):
        res = fit_lpm(grp, BASE_FORMULA)
        tables.append(amce_table(res, f"model:{model}"))

    pooled_base = fit_lpm(df, BASE_FORMULA + " + C(model)")
    tables.append(amce_table(pooled_base, "pooled"))
    pd.concat(tables).to_csv(out / f"amce_{args.arm}.csv", index=False)

    pooled_int = fit_lpm(df, INTERACT_FORMULA + " + C(model)")
    raw_p = {}
    test_rows = []
    for h, pattern in CONFIRMATORY.items():
        use = (pooled_int if int(h[1:h.index('_')]) >= FIRST_INTERACTION_H
               else pooled_base)
        wstat, p, k = joint_test(use, pattern)
        raw_p[h] = p
        test_rows.append(dict(hypothesis=h, wald=round(wstat, 2),
                              df=k, p_raw=p))
    adj = holm({k: v for k, v in raw_p.items() if not np.isnan(v)})
    for row in test_rows:
        row["p_holm"] = adj.get(row["hypothesis"], np.nan)
        row["significant"] = bool(row["p_holm"] < 0.05) if not np.isnan(
            row.get("p_holm", np.nan)) else False
    pd.DataFrame(test_rows).to_csv(out / "confirmatory_tests.csv", index=False)

    if args.arm == "main":
        intersectional_table(df).to_csv(out / "intersectional.csv", index=False)

    importance_shares(pooled_base).to_csv(out / "importance_shares.csv",
                                          index=False)

    print(f"[amce] arm={args.arm}: models={df['model'].nunique()}, "
          f"rows={len(df)}")
    print(pd.DataFrame(test_rows).to_string(index=False))


if __name__ == "__main__":
    main()
