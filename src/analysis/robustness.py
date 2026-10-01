"""Preregistered robustness battery.

Checks: template heterogeneity, temperature invariance (ratios), grounded
snippet vs card, field-order sensitivity, test-retest consistency,
plausibility-cell exclusion, name placebo (joint LR), missingness model.

Output: results/tables/robustness_*.csv

Run:  python -m src.analysis.robustness
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from src.design import load_config
from src.analysis.amce import load_choices, fit_lpm, BASE_FORMULA, amce_table

ROOT = Path(__file__).resolve().parents[1].parent

KEY_TERMS = ["C(rating)[T.4.7]", "C(volume)[T.400]", "recent", "responds",
             "network", "C(price)[T.190]", "telehealth",
             "C(experience)[T.28]", "female", "eth_black"]


def _amces(df: pd.DataFrame, label: str, formula: str = None) -> pd.DataFrame:
    f = formula or (BASE_FORMULA + (" + C(model)" if df["model"].nunique() > 1
                                    else ""))
    res = fit_lpm(df, f)
    t = amce_table(res, label)
    return t[t["term"].isin(KEY_TERMS)]


def template_heterogeneity(df: pd.DataFrame) -> pd.DataFrame:
    parts = [_amces(g, f"template:{t}") for t, g in df.groupby("template_id")]
    out = pd.concat(parts)
    spread = (out.groupby("term")["amce_pp"]
              .agg(["min", "max", "std"]).reset_index()
              .rename(columns={"min": "amce_min", "max": "amce_max",
                               "std": "amce_sd"}))
    return out, spread


def temperature_invariance(main: pd.DataFrame, t0: pd.DataFrame) -> pd.DataFrame:
    rows = []
    common = set(main["trial_id"]) & set(t0["trial_id"])
    m_sub = main[main["trial_id"].isin(common)]
    for label, d in [("temp_0.7", m_sub), ("temp_0.0", t0)]:
        a = _amces(d, label)
        price = a.loc[a["term"] == "C(price)[T.190]", "amce_pp"]
        denom = float(price.iloc[0]) if len(price) else np.nan
        for _, r in a.iterrows():
            rows.append(dict(condition=label, term=r["term"],
                             amce_pp=r["amce_pp"],
                             ratio_to_price=round(r["amce_pp"] / denom, 3)
                             if denom else np.nan))
    return pd.DataFrame(rows)


def format_comparisons(main: pd.DataFrame, other: pd.DataFrame,
                       label: str) -> pd.DataFrame:
    common = set(main["trial_id"]) & set(other["trial_id"])
    a = _amces(main[main["trial_id"].isin(common)], "card_choice_first")
    b = _amces(other, label)
    merged = a.merge(b, on="term", suffixes=("_main", f"_{label}"))
    merged["diff_pp"] = (merged[f"amce_pp_{label}"] - merged["amce_pp_main"]).round(2)
    return merged[["term", "amce_pp_main", f"amce_pp_{label}", "diff_pp"]]


def retest_consistency(retest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, tid), grp in retest[retest["chosen"] == 1].groupby(
            ["model", "trial_id"]):
        counts = grp["choice"].value_counts()
        n = counts.sum()
        p = counts / n
        rows.append(dict(model=model, trial_id=tid, n_reps=int(n),
                         modal_share=round(counts.iloc[0] / n, 3),
                         entropy=round(float(-(p * np.log(p)).sum()), 3)))
    per_set = pd.DataFrame(rows)
    summary = (per_set.groupby("model")[["modal_share", "entropy"]]
               .mean().round(3).reset_index())
    return per_set, summary


def plausibility_exclusion(df: pd.DataFrame) -> pd.DataFrame:
    implaus = (((df["price"] == 90) & (df["rating"] == 4.7) & (df["volume"] == 400))
               | ((df["price"] == 190) & (df["rating"] == 3.9) & (df["volume"] == 12)))
    bad_sets = df.loc[implaus, ["model", "trial_id"]].drop_duplicates()
    keep = df.merge(bad_sets, on=["model", "trial_id"], how="left",
                    indicator=True)
    keep = keep[keep["_merge"] == "left_only"].drop(columns="_merge")
    a = _amces(df, "full_sample")
    b = _amces(keep, "plausible_only")
    merged = a.merge(b, on="term", suffixes=("_full", "_plaus"))
    merged["diff_pp"] = (merged["amce_pp_plaus"] - merged["amce_pp_full"]).round(2)
    return merged[["term", "amce_pp_full", "amce_pp_plaus", "diff_pp"]]


def name_placebo(df: pd.DataFrame) -> dict:
    """Name-EXEMPLAR placebo: both models carry full gender x ethnicity cell
    fixed effects, so adding first-name dummies tests whether the specific
    exemplar chosen WITHIN a demographic cell moves choices (it should not).
    Rank-deficient overlap with the cell dummies is handled by pinv; df comes
    from the design-matrix rank difference."""
    cells = " + C(model) + C(gender):C(ethnicity)"
    full = smf.ols(BASE_FORMULA + cells + " + C(name_first)", data=df).fit()
    restr = smf.ols(BASE_FORMULA + cells, data=df).fit()
    k = full.df_model - restr.df_model
    lr = 2 * (full.llf - restr.llf)
    return dict(lr_stat=round(float(lr), 2), df=int(k),
                p_value=round(float(stats.chi2.sf(lr, k)), 4))


def missingness_model() -> pd.DataFrame:
    cfg = load_config()
    rows = []
    for path in (ROOT / cfg["paths"]["raw"]).glob("*.jsonl"):
        if path.stem.endswith("_pilot") or "judge" in path.stem:
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("arm") in ("main",):
                    rows.append(dict(model=rec["model"],
                                     trial_id=rec["trial_id"],
                                     fail=int(not rec["ok"])))
    if not rows:
        return pd.DataFrame()
    calls = pd.DataFrame(rows)
    trials = pd.read_parquet(ROOT / cfg["paths"]["design"] / "trials.parquet")
    calls = calls.merge(trials[["trial_id", "persona", "template_id"]],
                        on="trial_id")
    if calls["fail"].sum() == 0:
        return pd.DataFrame([dict(note="zero parse failures; model not fit")])
    res = smf.logit("fail ~ C(model) + C(persona) + C(template_id)",
                    data=calls).fit(disp=False)
    return pd.DataFrame(dict(term=res.params.index,
                             beta=np.round(res.params, 3),
                             p=np.round(res.pvalues, 4)))


def main() -> None:
    cfg = load_config()
    out = ROOT / cfg["paths"]["tables"]
    out.mkdir(parents=True, exist_ok=True)

    main_df = load_choices("main")

    per_template, spread = template_heterogeneity(main_df)
    per_template.to_csv(out / "robustness_templates.csv", index=False)
    spread.to_csv(out / "robustness_template_spread.csv", index=False)

    for arm, label in [("temp0", None), ("grounded", "grounded"),
                       ("field_order", "reason_first"),
                       ("rank_top3", "rank_top3"),
                       ("nopersona", "nopersona"),
                       ("card_reversed", "card_reversed")]:
        try:
            arm_df = load_choices(arm)
        except (FileNotFoundError, KeyError):
            continue
        if arm_df.empty:
            continue
        if arm == "temp0":
            temperature_invariance(main_df, arm_df).to_csv(
                out / "robustness_temperature.csv", index=False)
        else:
            format_comparisons(main_df, arm_df, label).to_csv(
                out / f"robustness_{arm}.csv", index=False)

    try:
        retest = load_choices("retest")
        if not retest.empty:
            per_set, summary = retest_consistency(retest)
            per_set.to_csv(out / "robustness_retest_sets.csv", index=False)
            summary.to_csv(out / "robustness_retest_summary.csv", index=False)
    except (FileNotFoundError, KeyError):
        pass

    plausibility_exclusion(main_df).to_csv(
        out / "robustness_plausibility.csv", index=False)

    pd.DataFrame([name_placebo(main_df)]).to_csv(
        out / "robustness_name_placebo.csv", index=False)

    mm = missingness_model()
    if not mm.empty:
        mm.to_csv(out / "robustness_missingness.csv", index=False)

    print("[robustness] battery complete; tables under results/tables/")


if __name__ == "__main__":
    main()
