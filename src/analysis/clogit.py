"""Secondary analysis: conditional (McFadden) logit per model + pooled FE.

Output: results/tables/clogit_{arm}.csv

Run:  python -m src.analysis.clogit [--arm main]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.discrete.conditional_models import ConditionalLogit

from src.design import load_config
from src.analysis.amce import load_choices

ROOT = Path(__file__).resolve().parents[1].parent

X_COLS = ["rating_43", "rating_47", "volume_85", "volume_400", "recent",
          "responds", "network", "price_140", "price_190", "telehealth",
          "exp_18", "exp_28", "female",
          "eth_black", "eth_hispanic", "eth_east_asian", "eth_south_asian",
          "slot_2", "slot_3", "slot_4", "slot_5"]


def design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    x["rating_43"] = (df["rating"] == 4.3).astype(float)
    x["rating_47"] = (df["rating"] == 4.7).astype(float)
    x["volume_85"] = (df["volume"] == 85).astype(float)
    x["volume_400"] = (df["volume"] == 400).astype(float)
    x["recent"] = df["recent"].astype(float)
    x["responds"] = df["responds"].astype(float)
    x["network"] = df["network"].astype(float)
    x["price_140"] = (df["price"] == 140).astype(float)
    x["price_190"] = (df["price"] == 190).astype(float)
    x["telehealth"] = df["telehealth"].astype(float)
    x["exp_18"] = (df["experience"] == 18).astype(float)
    x["exp_28"] = (df["experience"] == 28).astype(float)
    x["female"] = df["female"].astype(float)
    for eth in ("black", "hispanic", "east_asian", "south_asian"):
        x[f"eth_{eth}"] = df[f"eth_{eth}"].astype(float)
    for s in range(2, 6):
        x[f"slot_{s}"] = (df["slot"] == s).astype(float)
    return x


def fit_group(df: pd.DataFrame, group_col: str):
    """Fit conditional logit; groups = choice sets (or model x set when pooled)."""
    x = design_matrix(df)
    model = ConditionalLogit(df["chosen"].to_numpy(), x.to_numpy(),
                             groups=df[group_col].to_numpy())
    return model.fit(disp=False), x.columns.tolist()


def tidy(res, cols, scope: str) -> pd.DataFrame:
    return pd.DataFrame(dict(
        scope=scope, term=cols, beta=np.round(res.params, 4),
        se=np.round(res.bse, 4), odds_ratio=np.round(np.exp(res.params), 3),
        p=res.pvalues))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="main")
    args = ap.parse_args()

    cfg = load_config()
    df = load_choices(args.arm)
    # drop choice sets without exactly one chosen card (parse failure remnants)
    ok_sets = df.groupby(["model", "trial_id"])["chosen"].sum()
    ok_keys = ok_sets[ok_sets == 1].index
    df = df.set_index(["model", "trial_id"]).loc[ok_keys].reset_index()
    df["set_id"] = df["model"].astype(str) + "_" + df["trial_id"].astype(str)

    out = []
    for model, grp in df.groupby("model"):
        try:
            res, cols = fit_group(grp, "trial_id")
            out.append(tidy(res, cols, f"model:{model}"))
        except Exception as e:  # separation etc.
            print(f"[clogit] {model}: estimation failed ({e}); "
                  f"report AMCEs only for this scope")
    res, cols = fit_group(df, "set_id")
    out.append(tidy(res, cols, "pooled"))

    out_dir = ROOT / cfg["paths"]["tables"]
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(out).to_csv(out_dir / f"clogit_{args.arm}.csv", index=False)
    print(f"[clogit] arm={args.arm} written "
          f"({df['model'].nunique()} models, {len(df)} rows)")


if __name__ == "__main__":
    main()
