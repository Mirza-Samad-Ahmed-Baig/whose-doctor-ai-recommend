"""Revision-round analyses requested in peer review (IJMEDI-D-26-04763, R1).

These analyses were NOT in the prespecified plan; they are reported as
post hoc, reviewer-requested robustness analyses. They leave every
prespecified estimate unchanged and only re-read the same main-arm data.

  1. Model heterogeneity   per-model AMCEs + TOST, model x attribute Wald
                           tests, DerSimonian-Laird random-effects summary
                           with prediction intervals, leave-one-model-out
  2. Name exemplars        cluster-robust within-cell name placebo (first
                           names and surnames), names-as-units second-stage
                           estimates, two-way (choice set x first name)
                           clustering, leave-one-name-out ranges
  3. Fee nonlinearity      linearity test, segment slopes, scale-free
                           fee-range index, piecewise dollar conversions
  4. H13 family            Holm over H1-H13 (as implemented) vs H13 alone;
                           individual gender x ethnicity interaction terms
  5. Abstention            main-arm vs all-arm rates
  6. Perceived gender      gendered pronouns in stated reasons vs the
                           name-signaled gender (manipulation check)

Output: results/tables/revision/*.csv, results/figures/F8_heterogeneity.pdf,
        results/figures/F9_name_exemplars.pdf

Run:  python -m src.analysis.revision_r1
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from src.design import load_config, FIRST_NAMES, SURNAMES
from src.analysis.amce import (load_choices, fit_lpm, BASE_FORMULA,
                               INTERACT_FORMULA, CONFIRMATORY, joint_test,
                               holm, tost)

ROOT = Path(__file__).resolve().parents[1].parent
RNG_SEED = 20260928
N_DRAWS = 10_000

KEY_TERMS = {
    "C(rating)[T.4.7]": "Rating 4.7 (vs 3.9)",
    "C(price)[T.190]": "$190 fee (vs $90)",
    "C(volume)[T.400]": "400 reviews (vs 12)",
    "C(slot)[T.5]": "Position 5 (vs 1)",
    "female": "Female name (vs male)",
    "eth_black": "Black name (vs White)",
    "eth_hispanic": "Hispanic name (vs White)",
    "eth_east_asian": "East Asian name (vs White)",
    "eth_south_asian": "South Asian name (vs White)",
}
DEMO_TERMS = ["female", "eth_black", "eth_hispanic", "eth_east_asian",
              "eth_south_asian"]
FAMILIES = {  # attribute family -> regex on the base-model term name
    "rating": r"C\(rating\)\[T\.", "volume": r"C\(volume\)\[T\.",
    "recency": r"recent$", "response": r"responds$", "affiliation": r"network$",
    "fee": r"C\(price\)\[T\.", "telehealth": r"telehealth$",
    "experience": r"C\(experience\)\[T\.", "gender": r"female$",
    "ethnicity": r"eth_\w+$", "position": r"C\(slot\)\[T\.",
}
SESOI = 0.015
RHS = BASE_FORMULA.split("~", 1)[1].strip()
DEMO_RHS = " + female + eth_black + eth_hispanic + eth_east_asian + eth_south_asian"
NODEMO_RHS = RHS.replace(DEMO_RHS, "")
assert NODEMO_RHS != RHS, "demographic block not found in BASE_FORMULA"


def _out() -> Path:
    cfg = load_config()
    p = ROOT / cfg["paths"]["tables"] / "revision"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _fig_dir() -> Path:
    cfg = load_config()
    p = ROOT / cfg["paths"]["figures"]
    p.mkdir(parents=True, exist_ok=True)
    return p


def _row(res, term: str, **extra) -> dict:
    b, se = float(res.params[term]), float(res.bse[term])
    return dict(term=term, amce_pp=100 * b, se_pp=100 * se,
                ci_lo_pp=100 * (b - 1.96 * se), ci_hi_pp=100 * (b + 1.96 * se),
                p=float(res.pvalues[term]), tost_p=float(tost(b, se, SESOI)),
                **extra)


def _wald(res, R: np.ndarray) -> tuple[float, int, float]:
    """Cluster-robust Wald chi2 for R b = 0, rank-aware (pinv)."""
    b = res.params.to_numpy()
    V = res.cov_params().to_numpy()
    Rb = R @ b
    M = R @ V @ R.T
    k = int(np.linalg.matrix_rank(M))
    w = float(Rb @ np.linalg.pinv(M) @ Rb)
    return w, k, float(stats.chi2.sf(w, k))


# ---------------------------------------------------------------------------
# 1. Model heterogeneity
# ---------------------------------------------------------------------------

def dersimonian_laird(est: np.ndarray, se: np.ndarray) -> dict:
    w = 1 / se ** 2
    mu_fe = np.sum(w * est) / np.sum(w)
    q = float(np.sum(w * (est - mu_fe) ** 2))
    k = len(est)
    c = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (q - (k - 1)) / c)
    w_re = 1 / (se ** 2 + tau2)
    mu = float(np.sum(w_re * est) / np.sum(w_re))
    se_mu = float(np.sqrt(1 / np.sum(w_re)))
    i2 = max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0
    t = stats.t.ppf(0.975, k - 2)
    half = t * np.sqrt(tau2 + se_mu ** 2)
    return dict(k=k, re_mean=mu, re_se=se_mu, re_ci_lo=mu - 1.96 * se_mu,
                re_ci_hi=mu + 1.96 * se_mu, tau=float(np.sqrt(tau2)),
                q=q, q_p=float(stats.chi2.sf(q, k - 1)), i2=i2,
                pi_lo=mu - half, pi_hi=mu + half)


def model_heterogeneity(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    models = sorted(df["model"].unique())

    # (a) per-model estimates for every term, with TOST
    rows = []
    for m in models:
        res = fit_lpm(df[df["model"] == m], BASE_FORMULA)
        for term in res.params.index:
            if term == "Intercept":
                continue
            rows.append(_row(res, term, model=m))
    per_model = pd.DataFrame(rows)
    per_model.to_csv(out / "per_model_amce.csv", index=False)

    # (b) model x attribute heterogeneity: fully interacted pooled model,
    #     choice-set-clustered; test all model-interaction terms per family
    full = fit_lpm(df, f"chosen ~ C(model) * ({RHS})")
    het_rows = []
    for fam, pat in FAMILIES.items():
        # model ids contain ":" (llama3.1:8b), so split on the "]:" boundary
        names = [n for n in full.params.index
                 if n.startswith("C(model)[T.") and "]:" in n
                 and re.search(pat, n.split("]:", 1)[1])]
        R = np.zeros((len(names), len(full.params)))
        idx = list(full.params.index)
        for i, n in enumerate(names):
            R[i, idx.index(n)] = 1
        w, k, p = _wald(full, R)
        het_rows.append(dict(family=fam, wald=w, df=k, p=p))
    het = pd.DataFrame(het_rows)
    het["p_holm"] = het["family"].map(
        holm(dict(zip(het["family"], het["p"]))))
    het.to_csv(out / "model_heterogeneity_tests.csv", index=False)

    # (c) random-effects summary across models (approximate: models share
    #     stimuli, so per-model estimates are not fully independent)
    re_rows = []
    for term, label in KEY_TERMS.items():
        sub = per_model[per_model["term"] == term]
        d = dersimonian_laird(sub["amce_pp"].to_numpy(), sub["se_pp"].to_numpy())
        d.update(term=term, label=label,
                 min_model=sub.loc[sub["amce_pp"].idxmin(), "model"],
                 min_pp=sub["amce_pp"].min(),
                 max_model=sub.loc[sub["amce_pp"].idxmax(), "model"],
                 max_pp=sub["amce_pp"].max(),
                 n_sig_pos=int(((sub["ci_lo_pp"] > 0)).sum()),
                 n_sig_neg=int(((sub["ci_hi_pp"] < 0)).sum()),
                 n_equiv=int((sub["tost_p"] < 0.05).sum()))
        re_rows.append(d)
    re_df = pd.DataFrame(re_rows)
    re_df.to_csv(out / "random_effects_summary.csv", index=False)

    # (d) leave-one-model-out pooled estimates
    lomo = []
    for m in models:
        res = fit_lpm(df[df["model"] != m], BASE_FORMULA + " + C(model)")
        for term in KEY_TERMS:
            lomo.append(_row(res, term, dropped=m))
    lomo_df = pd.DataFrame(lomo)
    lomo_df.to_csv(out / "leave_one_model_out.csv", index=False)

    print("[revision] heterogeneity tests:\n", het.round(4).to_string(index=False))
    print(re_df[["label", "re_mean", "tau", "i2", "pi_lo", "pi_hi",
                 "n_sig_pos", "n_equiv"]].round(2).to_string(index=False))
    return per_model


# ---------------------------------------------------------------------------
# 2. Name exemplars
# ---------------------------------------------------------------------------

def _first_name_attrs() -> pd.DataFrame:
    rows = []
    for (g, e), names in FIRST_NAMES.items():
        for n in names:
            rows.append(dict(name_first=n, gender=g, ethnicity=e))
    return pd.DataFrame(rows)


def _level_coefs(res, var: str) -> pd.Series:
    pre = f"C({var})["
    s = res.params[[n for n in res.params.index if n.startswith(pre)]]
    s.index = [re.search(r"\[(?:T\.)?(.+)\]$", n).group(1) for n in s.index]
    return s


def name_exemplars(df: pd.DataFrame, out: Path) -> dict:
    res_out = {}
    cells = _first_name_attrs()

    # (a) cluster-robust within-cell first-name placebo: 40 name levels
    #     (no intercept), constraints = names equal within each g x e cell
    f_fn = f"chosen ~ 0 + C(name_first) + {NODEMO_RHS} + C(model)"
    res_fn = fit_lpm(df, f_fn)
    idx = list(res_fn.params.index)
    R = []
    for (g, e), names in FIRST_NAMES.items():
        base = idx.index(f"C(name_first)[{names[0]}]")
        for n in names[1:]:
            r = np.zeros(len(idx))
            r[idx.index(f"C(name_first)[{n}]")] = 1
            r[base] = -1
            R.append(r)
    w, k, p = _wald(res_fn, np.array(R))
    placebo = [dict(test="first-name exemplar (cluster-robust Wald)",
                    stat=w, df=k, p=p)]

    # surname placebo: within-ethnicity surname deviations
    d = df.copy()
    sn_cols = []
    for eth, names in SURNAMES.items():
        for n in names[1:]:
            c = f"sn_{n.lower()}"
            d[c] = (d["name_last"] == n).astype(int)
            sn_cols.append(c)
    res_sn = fit_lpm(d, BASE_FORMULA + " + C(model) + " + " + ".join(sn_cols))
    idx = list(res_sn.params.index)
    R = np.zeros((len(sn_cols), len(idx)))
    for i, c in enumerate(sn_cols):
        R[i, idx.index(c)] = 1
    w, k, p = _wald(res_sn, R)
    placebo.append(dict(test="surname exemplar within ethnicity "
                             "(cluster-robust Wald)", stat=w, df=k, p=p))
    pd.DataFrame(placebo).to_csv(out / "name_placebo_robust.csv", index=False)
    res_out["placebo"] = placebo

    # (b) names-as-units second stage: 40 first-name levels regressed on
    #     gender + ethnicity (df = 34); SEs reflect between-name variation
    lv = _level_coefs(res_fn, "name_first").rename("level").reset_index()
    lv = lv.rename(columns={"index": "name_first"}).merge(cells, on="name_first")
    lv_se = res_fn.bse[[f"C(name_first)[{n}]" for n in lv["name_first"]]]
    lv["level_se"] = lv_se.to_numpy()
    for e in ("black", "hispanic", "east_asian", "south_asian"):
        lv[f"eth_{e}"] = (lv["ethnicity"] == e).astype(int)
    lv["female"] = (lv["gender"] == "female").astype(int)
    s2 = smf.ols("level ~ female + eth_black + eth_hispanic + eth_east_asian"
                 " + eth_south_asian", data=lv).fit()
    s2_rows = []
    for term in DEMO_TERMS:
        b, se = float(s2.params[term]), float(s2.bse[term])
        t = stats.t.ppf(0.975, s2.df_resid)
        s2_rows.append(dict(term=term, unit="first name (n=40)",
                            amce_pp=100 * b, se_pp=100 * se,
                            ci_lo_pp=100 * (b - t * se),
                            ci_hi_pp=100 * (b + t * se),
                            p=float(s2.pvalues[term]), df_resid=s2.df_resid))

    # surname-level second stage for ethnicity (n=20, df=15)
    res_ln = fit_lpm(df, f"chosen ~ 0 + C(name_last) + {NODEMO_RHS}"
                         " + female + C(model)")
    sl = _level_coefs(res_ln, "name_last").rename("level").reset_index()
    sl = sl.rename(columns={"index": "name_last"})
    sl["ethnicity"] = sl["name_last"].map(
        {n: e for e, ns in SURNAMES.items() for n in ns})
    for e in ("black", "hispanic", "east_asian", "south_asian"):
        sl[f"eth_{e}"] = (sl["ethnicity"] == e).astype(int)
    s3 = smf.ols("level ~ eth_black + eth_hispanic + eth_east_asian"
                 " + eth_south_asian", data=sl).fit()
    for term in DEMO_TERMS[1:]:
        b, se = float(s3.params[term]), float(s3.bse[term])
        t = stats.t.ppf(0.975, s3.df_resid)
        s2_rows.append(dict(term=term, unit="surname (n=20)",
                            amce_pp=100 * b, se_pp=100 * se,
                            ci_lo_pp=100 * (b - t * se),
                            ci_hi_pp=100 * (b + t * se),
                            p=float(s3.pvalues[term]), df_resid=s3.df_resid))

    # (c) two-way clustering: choice set x first name
    base = BASE_FORMULA + " + C(model)"
    groups = np.column_stack([pd.factorize(df["trial_id"])[0],
                              pd.factorize(df["name_first"])[0]])
    res_2w = smf.ols(base, data=df).fit(cov_type="cluster",
                                        cov_kwds={"groups": groups})
    res_1w = fit_lpm(df, base)
    for term in DEMO_TERMS:
        for lab, r in (("choice set (as published)", res_1w),
                       ("two-way: choice set x first name", res_2w)):
            b, se = float(r.params[term]), float(r.bse[term])
            s2_rows.append(dict(term=term, unit=lab, amce_pp=100 * b,
                                se_pp=100 * se, ci_lo_pp=100 * (b - 1.96 * se),
                                ci_hi_pp=100 * (b + 1.96 * se),
                                p=float(r.pvalues[term]), df_resid=np.nan))
    names_inf = pd.DataFrame(s2_rows)
    names_inf.to_csv(out / "name_level_inference.csv", index=False)

    # (d) leave-one-name-out: drop every choice set containing the name
    lono = []
    for var, pool in (("name_first", sorted(cells["name_first"])),
                      ("name_last", sorted(sum(SURNAMES.values(), [])))):
        for n in pool:
            drop = df.loc[df[var] == n, ["model", "trial_id"]].drop_duplicates()
            keep = df.merge(drop, on=["model", "trial_id"], how="left",
                            indicator=True)
            keep = keep[keep["_merge"] == "left_only"]
            r = fit_lpm(keep, base)
            for term in DEMO_TERMS:
                lono.append(dict(dropped_var=var, dropped=n, term=term,
                                 amce_pp=100 * float(r.params[term]),
                                 p=float(r.pvalues[term])))
    lono_df = pd.DataFrame(lono)
    lono_df.to_csv(out / "leave_one_name_out.csv", index=False)
    lono_sum = (lono_df.groupby(["dropped_var", "term"])
                .agg(min_pp=("amce_pp", "min"), max_pp=("amce_pp", "max"),
                     max_p=("p", "max")).reset_index())
    lono_sum.to_csv(out / "leave_one_name_out_summary.csv", index=False)

    # first-name effects relative to the grand mean of all 40 names, with
    # within-cell deviations for the exemplar figure
    lv["effect_pp"] = 100 * (lv["level"] - lv["level"].mean())
    lv["se_pp"] = 100 * lv["level_se"]
    lv["cell_mean_pp"] = lv.groupby(["gender", "ethnicity"])["effect_pp"].transform("mean")
    lv.to_csv(out / "first_name_effects.csv", index=False)
    sl["effect_pp"] = 100 * (sl["level"] - sl["level"].mean())
    sl.to_csv(out / "surname_effects.csv", index=False)

    print("[revision] name placebo:\n", pd.DataFrame(placebo).round(4).to_string(index=False))
    print(names_inf.round(3).to_string(index=False))
    print(lono_sum.round(3).to_string(index=False))
    res_out["first_names"] = lv
    return res_out


# ---------------------------------------------------------------------------
# 3. Fee nonlinearity and scale-free trade-off indices
# ---------------------------------------------------------------------------

TRADE_TERMS = {
    "C(rating)[T.4.3]": "Rating 4.3 (vs 3.9)",
    "C(rating)[T.4.7]": "Rating 4.7 (vs 3.9)",
    "C(volume)[T.85]": "85 reviews (vs 12)",
    "C(volume)[T.400]": "400 reviews (vs 12)",
    "recent": "Recent review (vs 11 mo.)",
    "responds": "Responds to patient feedback",
    "network": "Hospital-affiliated (vs independent)",
    "telehealth": "Telehealth available",
    "C(experience)[T.18]": "18 yrs in practice (vs 8)",
    "C(experience)[T.28]": "28 yrs in practice (vs 8)",
    "female": "Female name (vs male)",
    "eth_black": "Black name (vs White)",
    "eth_hispanic": "Hispanic name (vs White)",
    "eth_east_asian": "East Asian name (vs White)",
    "eth_south_asian": "South Asian name (vs White)",
    "POS1": "First-listed (vs avg of slots 2-5)",
}


def _piecewise_dollars(effect: np.ndarray, d140: np.ndarray,
                       d190: np.ndarray) -> np.ndarray:
    """Fee increase above $90 whose (piecewise-linear) penalty equals
    `effect`; NaN where |effect| exceeds the tested $90->$190 penalty."""
    a = np.abs(effect)
    out = np.where(a <= d140, 50 * a / d140,
                   50 + 50 * (a - d140) / (d190 - d140))
    out = np.where(a > d190, np.nan, out)
    return np.sign(effect) * out


def fee_nonlinearity(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    res = fit_lpm(df, BASE_FORMULA + " + C(model)")
    idx = list(res.params.index)
    b140, b190 = res.params["C(price)[T.140]"], res.params["C(price)[T.190]"]
    R = np.zeros((1, len(idx)))
    R[0, idx.index("C(price)[T.190]")] = 1
    R[0, idx.index("C(price)[T.140]")] = -2
    w, k, p = _wald(res, R)
    lin = dict(amce_140_pp=100 * b140, amce_190_pp=100 * b190,
               slope_90_140_pp_per_usd=-100 * b140 / 50,
               slope_140_190_pp_per_usd=-100 * (b190 - b140) / 50,
               linear_slope_pp_per_usd=-100 * b190 / 100,
               linearity_wald=w, linearity_df=k, linearity_p=p)
    pd.DataFrame([lin]).to_csv(out / "fee_linearity.csv", index=False)

    rng = np.random.default_rng(RNG_SEED)
    draws = rng.multivariate_normal(res.params.to_numpy(),
                                    res.cov_params().to_numpy(), size=N_DRAWS)
    col = lambda n: draws[:, idx.index(n)]
    d140, d190 = -col("C(price)[T.140]"), -col("C(price)[T.190]")
    slots = [n for n in idx if n.startswith("C(slot)")]
    pos1 = -np.mean([col(n) for n in slots], axis=0)

    rows = []
    for term, label in TRADE_TERMS.items():
        eff = pos1 if term == "POS1" else col(term)
        index = eff / d190
        lin_usd = eff / (d190 / 100)
        seg1_usd = eff / (d140 / 50)
        seg2_usd = eff / ((d190 - d140) / 50)
        pw = _piecewise_dollars(eff, d140, d190)
        q = lambda x: np.nanpercentile(x, [50, 2.5, 97.5])
        pt = float(np.median(eff))
        beyond = float(np.mean(np.abs(eff) > d190))
        rows.append(dict(
            term=term, label=label, amce_pp=100 * pt,
            fee_index=q(index)[0], fee_index_lo=q(index)[1],
            fee_index_hi=q(index)[2],
            usd_linear=q(lin_usd)[0], usd_linear_lo=q(lin_usd)[1],
            usd_linear_hi=q(lin_usd)[2],
            usd_lower_segment=q(seg1_usd)[0],
            usd_upper_segment=q(seg2_usd)[0],
            usd_piecewise_from90=(np.nan if beyond > 0.5 else q(pw)[0]),
            share_draws_beyond_tested_range=beyond))
    tr = pd.DataFrame(rows)
    tr.to_csv(out / "fee_index.csv", index=False)
    print("[revision] fee linearity:", {k_: round(v, 4) for k_, v in lin.items()})
    print(tr.round(3).to_string(index=False))
    return tr


# ---------------------------------------------------------------------------
# 4. H13 family sensitivity
# ---------------------------------------------------------------------------

def h13_sensitivity(df: pd.DataFrame, out: Path) -> None:
    base = fit_lpm(df, BASE_FORMULA + " + C(model)")
    inter = fit_lpm(df, INTERACT_FORMULA + " + C(model)")
    raw = {}
    for h, pat in CONFIRMATORY.items():
        use = inter if int(h[1:h.index("_")]) >= 11 else base
        raw[h] = joint_test(use, pat)[1]
    h13 = holm(raw)
    h12 = holm({k: v for k, v in raw.items() if not k.startswith("H13")})
    rows = [dict(hypothesis=h, p_raw=raw[h], p_holm_H1_H13=h13[h],
                 p_holm_H1_H12=h12.get(h, np.nan)) for h in raw]
    pd.DataFrame(rows).to_csv(out / "holm_family_sensitivity.csv", index=False)
    terms = [n for n in inter.params.index if n.startswith("female:eth_")]
    pd.DataFrame([_row(inter, t) for t in terms]).to_csv(
        out / "h13_interaction_terms.csv", index=False)
    print("[revision] H13 raw p = %.4f; Holm(H1-H13) = %.4f"
          % (raw["H13_intersectional"], h13["H13_intersectional"]))


# ---------------------------------------------------------------------------
# 5. Abstention: main arm vs all arms
# ---------------------------------------------------------------------------

def abstention_rates(out: Path) -> None:
    cfg = load_config()
    s = pd.read_csv(ROOT / cfg["paths"]["tables"] / "refusals_summary.csv")
    s["n_abstain"] = s["n"] * s["abstain_rate"]
    s["n_demo"] = s["n"] * s["demo_aware_rate"]
    rows = []
    for label, sub in (("main arm", s[s["arm"] == "main"]), ("all arms", s)):
        rows.append(dict(scope=label, n=int(sub["n"].sum()),
                         abstain_pct=100 * sub["n_abstain"].sum() / sub["n"].sum(),
                         demo_aware_pct=100 * sub["n_demo"].sum() / sub["n"].sum()))
    pd.DataFrame(rows).to_csv(out / "abstention_scope.csv", index=False)
    print("[revision] abstention:\n", pd.DataFrame(rows).round(3).to_string(index=False))


# ---------------------------------------------------------------------------
# 6. Perceived gender: pronouns in stated reasons as a manipulation check
# ---------------------------------------------------------------------------

FEM_PRON = r"\b(?:she|her|hers|herself)\b"
MASC_PRON = r"\b(?:he|him|his|himself)\b"


def pronoun_check(df: pd.DataFrame, out: Path) -> None:
    """Share of reasons that refer to the chosen physician with a gendered
    pronoun, and agreement of that pronoun with the name-signaled gender.
    Reasons mentioning both pronoun sets are counted as ambiguous."""
    ch = df[(df["chosen"] == 1) & df["reason"].notna()].drop_duplicates(
        ["model", "trial_id"]).copy()
    txt = ch["reason"].astype(str)
    fem = txt.str.contains(FEM_PRON, case=False, regex=True)
    masc = txt.str.contains(MASC_PRON, case=False, regex=True)
    ch["pron"] = np.select([fem & ~masc, masc & ~fem, fem & masc],
                           ["female", "male", "both"], "none")
    ch["match"] = ch["pron"] == ch["gender"]
    one = ch[ch["pron"].isin(["female", "male"])]

    def summarise(g: pd.DataFrame) -> pd.Series:
        o = g[g["pron"].isin(["female", "male"])]
        return pd.Series(dict(
            n_reasons=len(g), pronoun_share=float((g["pron"] != "none").mean()),
            n_single_pronoun=len(o),
            agree_share=float(o["match"].mean()) if len(o) else np.nan))

    by_model = ch.groupby("model").apply(summarise, include_groups=False)
    pooled = summarise(ch).to_frame("pooled").T
    pd.concat([by_model, pooled]).to_csv(out / "pronoun_check_by_model.csv")
    by_name = (one.groupby(["name_first", "gender", "ethnicity"])
               .agg(n=("match", "size"), agree_share=("match", "mean"))
               .reset_index().sort_values("agree_share"))
    by_name.to_csv(out / "pronoun_check_by_first_name.csv", index=False)
    by_cell = (one.groupby(["gender", "ethnicity"])
               .agg(n=("match", "size"), agree_share=("match", "mean"))
               .reset_index())
    by_cell.to_csv(out / "pronoun_check_by_cell.csv", index=False)
    print("[revision] pronoun check:\n",
          pd.concat([by_model, pooled]).round(3).to_string())
    print(by_name.head(8).round(3).to_string(index=False))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figures(per_model: pd.DataFrame, first_names: pd.DataFrame,
            out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.analysis.figures import PALETTE, _grid  # shared house style

    fig_dir = _fig_dir()
    re_df = pd.read_csv(out / "random_effects_summary.csv").set_index("term")

    # F8: per-model demographic AMCEs with random-effects prediction interval
    terms = DEMO_TERMS
    models = sorted(per_model["model"].unique())
    short = {m: m.replace("mistral:7b-instruct-q4_K_M", "mistral:7b")
             for m in models}
    fig, axes = plt.subplots(1, len(terms), figsize=(10.5, 3.3), sharey=True)
    for ax, term in zip(axes, terms):
        sub = per_model[per_model["term"] == term].set_index("model").loc[models]
        y = np.arange(len(models))
        ax.axvspan(-100 * SESOI, 100 * SESOI, color="#EEEEEE", zorder=0)
        ax.axvline(0, color="#999999", lw=0.8)
        ax.errorbar(sub["amce_pp"], y,
                    xerr=[sub["amce_pp"] - sub["ci_lo_pp"],
                          sub["ci_hi_pp"] - sub["amce_pp"]],
                    fmt="o", ms=4, color=PALETTE[0], ecolor=PALETTE[0],
                    elinewidth=1, capsize=0)
        r = re_df.loc[term]
        yy = len(models) + 0.6
        ax.errorbar([r["re_mean"]], [yy],
                    xerr=[[r["re_mean"] - r["re_ci_lo"]],
                          [r["re_ci_hi"] - r["re_mean"]]],
                    fmt="D", ms=5, color=PALETTE[5], elinewidth=1.4)
        ax.plot([r["pi_lo"], r["pi_hi"]], [yy + 0.45] * 2, color=PALETTE[5],
                lw=1, ls="--")
        ax.set_title(KEY_TERMS[term].replace(" (vs male)", "\n(vs male)")
                     .replace(" (vs White)", "\n(vs White)"), fontsize=9)
        ax.set_xlabel("AMCE (pp)")
        _grid(ax, "x")
    axes[0].set_yticks(list(range(len(models))) + [len(models) + 0.6])
    axes[0].set_yticklabels([short[m] for m in models] + ["Random-effects\nmean"])
    axes[0].invert_yaxis()
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"F8_heterogeneity.{ext}")
    plt.close(fig)

    # F9: first-name exemplar effects grouped by gender x ethnicity cell
    fn = first_names.copy()
    order = [(g, e) for e in ("white", "black", "hispanic", "east_asian",
                              "south_asian") for g in ("male", "female")]
    fig, ax = plt.subplots(figsize=(10.5, 3.2))
    x, ticks, labels = 0, [], []
    for g, e in order:
        sub = fn[(fn["gender"] == g) & (fn["ethnicity"] == e)].sort_values("effect_pp")
        xs = np.arange(x, x + len(sub))
        col = PALETTE[3] if g == "female" else PALETTE[0]
        ax.errorbar(xs, sub["effect_pp"], yerr=1.96 * sub["se_pp"], fmt="o",
                    ms=3.5, color=col, elinewidth=0.9)
        ax.hlines(sub["cell_mean_pp"].iloc[0], xs[0] - 0.3, xs[-1] + 0.3,
                  color=col, lw=1.6)
        for xi, n in zip(xs, sub["name_first"]):
            ax.annotate(n, (xi, -12.6), rotation=90, ha="center", va="bottom",
                        fontsize=6.5, color="#444444")
        ticks.append(xs.mean())
        labels.append(f"{g[0].upper()}·{e.replace('_', ' ').title()}")
        x += len(sub) + 1
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("First-name effect (pp,\nvs mean of all names)")
    ax.set_ylim(-13, None)
    _grid(ax, "y")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"F9_name_exemplars.{ext}")
    plt.close(fig)
    print("[revision] wrote F8_heterogeneity, F9_name_exemplars")


def main() -> None:
    out = _out()
    df = load_choices("main")
    per_model = model_heterogeneity(df, out)
    names = name_exemplars(df, out)
    fee_nonlinearity(df, out)
    h13_sensitivity(df, out)
    abstention_rates(out)
    pronoun_check(df, out)
    figures(per_model, names["first_names"], out)
    print(f"[revision] all outputs under {out}")


if __name__ == "__main__":
    main()
