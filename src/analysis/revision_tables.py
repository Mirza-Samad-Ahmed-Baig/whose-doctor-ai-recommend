"""LaTeX tables for the reviewer-requested (post hoc) revision analyses.

Reads results/tables/revision/*.csv written by revision_r1 and writes table
floats to results/tables/revision/latex/.

Run:  python -m src.analysis.revision_tables
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.design import load_config

ROOT = Path(__file__).resolve().parents[1].parent

MODELS = ["gemma3:4b", "gpt-4o-mini", "llama3.1:8b", "llama3.2:3b",
          "mistral:7b-instruct-q4_K_M", "phi3:mini", "qwen2.5:3b"]
SHORT = {"mistral:7b-instruct-q4_K_M": "mistral:7b"}
HET_TERMS = [("C(rating)[T.4.7]", "Rating\\\\4.7"),
             ("C(price)[T.190]", "Fee\\\\\\$190"),
             ("C(slot)[T.5]", "Position\\\\5"),
             ("female", "Female-\\\\signaled"),
             ("eth_black", "Black-\\\\signaled"),
             ("eth_hispanic", "Hispanic-\\\\signaled"),
             ("eth_east_asian", "E.~Asian-\\\\signaled"),
             ("eth_south_asian", "S.~Asian-\\\\signaled")]
FAMILY_OF = {"C(rating)[T.4.7]": "rating", "C(price)[T.190]": "fee",
             "C(slot)[T.5]": "position", "female": "gender",
             "eth_black": "ethnicity", "eth_hispanic": "ethnicity",
             "eth_east_asian": "ethnicity", "eth_south_asian": "ethnicity"}
DEMO = [("female", "Female-signaled (vs male)"),
        ("eth_black", "Black-signaled (vs White)"),
        ("eth_hispanic", "Hispanic-signaled (vs White)"),
        ("eth_east_asian", "East Asian-signaled (vs White)"),
        ("eth_south_asian", "South Asian-signaled (vs White)")]
ALL_TERMS = [
    ("C(rating)[T.4.3]", "Rating 4.3 (vs 3.9)"),
    ("C(rating)[T.4.7]", "Rating 4.7 (vs 3.9)"),
    ("C(volume)[T.85]", "85 reviews (vs 12)"),
    ("C(volume)[T.400]", "400 reviews (vs 12)"),
    ("recent", "Recent review (vs 11 mo.)"),
    ("responds", "Responds to feedback"),
    ("network", "Hospital-affiliated"),
    ("C(price)[T.140]", "\\$140 fee (vs \\$90)"),
    ("C(price)[T.190]", "\\$190 fee (vs \\$90)"),
    ("telehealth", "Telehealth available"),
    ("C(experience)[T.18]", "18 yrs (vs 8)"),
    ("C(experience)[T.28]", "28 yrs (vs 8)"),
    ("female", "Female-signaled name"),
    ("eth_black", "Black-signaled name"),
    ("eth_hispanic", "Hispanic-signaled name"),
    ("eth_east_asian", "East Asian-signaled name"),
    ("eth_south_asian", "South Asian-signaled name"),
    ("C(slot)[T.2]", "Position 2 (vs 1)"),
    ("C(slot)[T.3]", "Position 3 (vs 1)"),
    ("C(slot)[T.4]", "Position 4 (vs 1)"),
    ("C(slot)[T.5]", "Position 5 (vs 1)"),
]


def _p(p: float) -> str:
    if np.isnan(p):
        return "--"
    if p < 0.001:
        return "<.001"
    return f"{p:.3f}".lstrip("0")


def _f1(x: float) -> str:
    """One decimal, without a negative sign on values that round to zero."""
    s = f"{x:.1f}"
    return "0.0" if s == "-0.0" else s


def _ci(lo: float, hi: float, nd: int = 1) -> str:
    return f"[{lo:.{nd}f}, {hi:.{nd}f}]"


def _table(body: str, caption: str, label: str, star: bool = False,
           resize: bool = False) -> str:
    """Table float; a trailing note (after "\\\\[2pt]") stays outside the
    resizebox so it is typeset at normal size."""
    env = "table*" if star else "table"
    body, sep, note = body.partition("\\\\[2pt]\n")
    inner = (f"\\resizebox{{\\ifdim\\width>\\linewidth \\linewidth\\else \\width\\fi}}{{!}}{{%\n{body}}}" if resize else body)
    tail = f"\n\n\\smallskip\n{note}" if sep else ""
    return (f"\\begin{{{env}}}[tbp]\n\\centering\n\\caption{{{caption}}}\n"
            f"\\label{{tab:{label}}}\n{inner}{tail}\n\\end{{{env}}}\n")


def _write(out: Path, name: str, tex: str) -> None:
    (out / name).write_text(tex, encoding="utf-8")
    print(f"[revision_tables] wrote {name}")


def tab_heterogeneity(src: Path, out: Path) -> None:
    pm = pd.read_csv(src / "per_model_amce.csv").set_index(["model", "term"])
    re_ = pd.read_csv(src / "random_effects_summary.csv").set_index("term")
    het = pd.read_csv(src / "model_heterogeneity_tests.csv").set_index("family")
    head = " & ".join(f"\\shortstack{{{lab}}}" for _, lab in HET_TERMS)
    rows = []
    for m in MODELS:
        cells = []
        for t, _ in HET_TERMS:
            r = pm.loc[(m, t)]
            mark = ""
            if r["ci_lo_pp"] > 0 or r["ci_hi_pp"] < 0:
                mark = "$^{*}$"
            elif r["tost_p"] < 0.05:
                mark = "$^{\\dagger}$"
            cells.append(f"{r['amce_pp']:.1f}{mark}")
        rows.append(f"{SHORT.get(m, m)} & " + " & ".join(cells) + " \\\\")
    rows.append("\\midrule")
    rows.append("Random-effects mean & " + " & ".join(
        f"{re_.loc[t, 're_mean']:.1f}" for t, _ in HET_TERMS) + " \\\\")
    rows.append("\\quad 95\\% prediction interval & " + " & ".join(
        f"\\footnotesize{_ci(re_.loc[t, 'pi_lo'], re_.loc[t, 'pi_hi'])}"
        for t, _ in HET_TERMS) + " \\\\")
    rows.append("\\quad $I^2$ & " + " & ".join(
        f"{100 * re_.loc[t, 'i2']:.0f}\\%" for t, _ in HET_TERMS) + " \\\\")
    fams = []
    for fam in dict.fromkeys(FAMILY_OF[t] for t, _ in HET_TERMS):
        h = het.loc[fam]
        span = sum(FAMILY_OF[t] == fam for t, _ in HET_TERMS)
        cell = f"\\footnotesize $\\chi^2_{{{int(h['df'])}}}={h['wald']:.0f}$"
        fams.append(cell if span == 1 else f"\\multicolumn{{{span}}}{{c}}{{{cell}}}")
    rows.append("Model$\\times$attribute Wald & " + " & ".join(fams) + " \\\\")
    body = ("\\begin{tabular}{l" + "r" * len(HET_TERMS) + "}\n\\toprule\n"
            f"Model & {head} \\\\\n\\midrule\n" + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n")
    note = ("\\\\[2pt]\n\\parbox{\\textwidth}{\\footnotesize\\textit{Note.} "
            "Post hoc (reviewer-requested) analysis. Entries are per-model "
            "AMCEs in percentage points from separate linear probability "
            "models (main arm, choice-set--clustered SEs). $^{*}$95\\% CI "
            "excludes zero; $^{\\dagger}$statistically equivalent to zero "
            "within $\\pm$1.5 pp (TOST $p<.05$). Random-effects summaries "
            "use DerSimonian--Laird weights across the seven models; the "
            "prediction interval ($t_{5}$) indicates the range expected for "
            "a further model resembling this panel and is approximate "
            "because all models saw the same stimuli. The Wald row tests "
            "equality of the whole attribute family across models (ethnicity: "
            "all four contrasts jointly); every test has $P<.001$.}")
    _write(out, "tab_heterogeneity.tex", _table(
        body + note, "Between-model heterogeneity in selected AMCEs.",
        "heterogeneity", resize=True))


def tab_names(src: Path, out: Path) -> None:
    ni = pd.read_csv(src / "name_level_inference.csv")
    lono = pd.read_csv(src / "leave_one_name_out_summary.csv")
    plc = pd.read_csv(src / "name_placebo_robust.csv")
    specs = [("choice set (as published)", "Choice-set clustered (as submitted)"),
             ("two-way: choice set x first name",
              "Two-way clustered (set $\\times$ first name)"),
             ("first name (n=40)", "First names as units ($n=40$)"),
             ("surname (n=20)", "Surnames as units ($n=20$)")]
    rows = []
    for key, lab in specs:
        cells = []
        for t, _ in DEMO:
            r = ni[(ni["unit"] == key) & (ni["term"] == t)]
            if r.empty:
                cells.append("--")
                continue
            r = r.iloc[0]
            cells.append(f"{r['amce_pp']:.1f} \\footnotesize"
                         f"{_ci(r['ci_lo_pp'], r['ci_hi_pp'])}")
        rows.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    for var, lab in (("name_first", "Leave-one-first-name-out range"),
                     ("name_last", "Leave-one-surname-out range")):
        cells = []
        for t, _ in DEMO:
            r = lono[(lono["dropped_var"] == var) & (lono["term"] == t)].iloc[0]
            cells.append(f"{r['min_pp']:.1f} to {r['max_pp']:.1f}")
        rows.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    head = "Female & Black & Hispanic & East Asian & South Asian"
    body = ("\\begin{tabular}{l" + "c" * len(DEMO) + "}\n\\toprule\n"
            f"Unit of inference & {head} \\\\\n\\midrule\n" + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n")
    fn, sn = plc.iloc[0], plc.iloc[1]
    note = ("\\\\[2pt]\n\\parbox{\\textwidth}{\\footnotesize\\textit{Note.} "
            "Columns are name-signaled contrasts (female vs male; each "
            "ethnicity vs White). "
            "Post hoc (reviewer-requested) analysis of the main arm, pooled "
            "with model fixed effects. Point estimates in pp with 95\\% "
            "intervals. `Names as units' fits one coefficient per first name "
            "(or surname) and regresses these 40 (20) name-level estimates "
            "on the category indicators, so uncertainty reflects variation "
            "between names within a category. Leave-one-out rows drop every "
            "choice set containing the given name. Within-category placebo "
            "(cluster-robust Wald): first names "
            f"$\\chi^2_{{{int(fn['df'])}}}={fn['stat']:.1f}$, $P{_p(fn['p']) if _p(fn['p']).startswith('<') else '=' + _p(fn['p'])}$; "
            f"surnames $\\chi^2_{{{int(sn['df'])}}}={sn['stat']:.1f}$, "
            f"$P={_p(sn['p'])}$.}}")
    _write(out, "tab_names.tex", _table(
        body + note, "Name-signaled category effects under alternative units "
        "of inference.", "names", resize=True))


def tab_feeindex(src: Path, out: Path) -> None:
    fi = pd.read_csv(src / "fee_index.csv")
    lin = pd.read_csv(src / "fee_linearity.csv").iloc[0]
    rows = []
    for _, r in fi.iterrows():
        beyond = r["share_draws_beyond_tested_range"] > 0.5
        pw = ("beyond range" if beyond else f"{r['usd_lower_segment']:.0f}"
              f"--{r['usd_upper_segment']:.0f}")
        lab = r["label"].replace(" name (vs", "-signaled name (vs").replace("$", "\\$").replace("First-listed (vs avg of slots 2-5)",
                                                     "First-listed (vs avg.\\ slots 2--5)")
        rows.append(f"{lab} & {r['amce_pp']:.1f} & {r['fee_index']:.2f} "
                    f"{_ci(r['fee_index_lo'], r['fee_index_hi'], 2)} & "
                    f"{r['usd_linear']:.0f} & {pw} \\\\")
    body = ("\\begin{tabular}{lrcrc}\n\\toprule\n"
            "Attribute & AMCE (pp) & Fee-range index [95\\% CI] & "
            "\\shortstack{Linearized\\\\\\$ (as submitted)} & "
            "\\shortstack{Segment-specific\\\\\\$ range} \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    note = ("\\\\[2pt]\n\\parbox{\\textwidth}{\\footnotesize\\textit{Note.} "
            "Fee-range index = AMCE $\\div$ absolute AMCE of \\$190 vs \\$90 "
            f"({abs(lin['amce_190_pp']):.1f} pp), with Krinsky--Robb 95\\% "
            "intervals (10{,}000 draws); an index of 0.13 means the effect "
            "equals 13\\% of the penalty for moving across the full tested fee "
            "range. It is the prespecified fee-equivalent divided by \\$100 and "
            "is a within-experiment trade-off ratio, not a willingness to pay. "
            "The fee response is not linear (\\$90$\\to$\\$140: "
            f"{lin['slope_90_140_pp_per_usd']:.2f} pp per \\$; \\$140$\\to$\\$190: "
            f"{lin['slope_140_190_pp_per_usd']:.2f} pp per \\$; linearity Wald "
            f"$\\chi^2_1={lin['linearity_wald']:.0f}$, $P<.001$), so dollar "
            "conversions depend on the segment used: the last column converts "
            "with the lower and upper segment slopes. Effects larger than the "
            "whole tested fee penalty (index $>1$) have no within-range dollar "
            "equivalent.}")
    _write(out, "tab_feeindex.tex", _table(
        body + note, "Fee-scaled trade-off indices (pooled, main arm).",
        "feeindex", resize=True))


def tab_per_model_full(src: Path, out: Path) -> None:
    pm = pd.read_csv(src / "per_model_amce.csv").set_index(["model", "term"])
    head = " & ".join(SHORT.get(m, m) for m in MODELS)
    rows = []
    for t, lab in ALL_TERMS:
        cells = []
        for m in MODELS:
            r = pm.loc[(m, t)]
            cells.append(f"{_f1(r['amce_pp'])} ({r['se_pp']:.1f})")
        rows.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    body = ("\\begin{tabular}{l" + "r" * len(MODELS) + "}\n\\toprule\n"
            f"Term & {head} \\\\\n\\midrule\n" + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n")
    note = ("\\\\[2pt]\n\\parbox{\\textwidth}{\\footnotesize\\textit{Note.} "
            "AMCE in pp (choice-set--clustered SE), separate linear "
            "probability model per model, main arm.}")
    _write(out, "tabS_per_model_amce.tex", _table(
        body + note, "Per-model AMCEs for every attribute level.",
        "S_per_model", resize=True))


def tab_het_tests(src: Path, out: Path) -> None:
    het = pd.read_csv(src / "model_heterogeneity_tests.csv")
    rows = [f"{r['family']} & {r['wald']:.1f} & {int(r['df'])} & {_p(r['p'])} "
            f"& {_p(r['p_holm'])} \\\\" for _, r in het.iterrows()]
    body = ("\\begin{tabular}{lrrrr}\n\\toprule\n"
            "Attribute family & Wald $\\chi^2$ & df & $P$ & $P$ (Holm) \\\\\n"
            "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    note = ("\\\\[2pt]\n\\parbox{0.8\\textwidth}{\\footnotesize\\textit{Note.} "
            "Joint cluster-robust Wald tests that all model$\\times$attribute "
            "interaction terms of a family are zero, from a fully interacted "
            "pooled linear probability model (main arm).}")
    _write(out, "tabS_het_tests.tex", _table(
        body + note, "Tests of between-model heterogeneity by attribute family.",
        "S_het_tests"))


def tab_pronouns(src: Path, out: Path) -> None:
    bm = pd.read_csv(src / "pronoun_check_by_model.csv", index_col=0)
    bn = pd.read_csv(src / "pronoun_check_by_first_name.csv")
    rows = []
    for m in MODELS + ["pooled"]:
        r = bm.loc[m]
        name = "Pooled" if m == "pooled" else SHORT.get(m, m)
        if m == "pooled":
            rows.append("\\midrule")
        rows.append(f"{name} & {int(r['n_reasons']):,} & "
                    f"{100 * r['pronoun_share']:.1f} & {int(r['n_single_pronoun']):,}"
                    f" & {100 * r['agree_share']:.1f} \\\\".replace(",", "{,}"))
    body = ("\\begin{tabular}{lrrrr}\n\\toprule\n"
            "Model & Reasons & \\shortstack{Gendered\\\\pronoun (\\%)} & "
            "\\shortstack{Single-gender\\\\pronoun ($n$)} & "
            "\\shortstack{Agrees with\\\\name signal (\\%)} \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    low = bn[bn["agree_share"] < 0.95].sort_values("agree_share")
    low_txt = "; ".join(f"{r['name_first']} ({100 * r['agree_share']:.0f}\\%, "
                        f"$n={int(r['n'])}$)" for _, r in low.iterrows())
    note = ("\\\\[2pt]\n\\parbox{0.9\\textwidth}{\\footnotesize\\textit{Note.} "
            "Main-arm stated reasons for the chosen physician. A reason "
            "counts as using a gendered pronoun if it contains she/her/hers/"
            "herself or he/him/his/himself; agreement is computed over "
            "reasons using only one of the two sets. First names with "
            f"agreement below 95\\%: {low_txt}.}}")
    _write(out, "tabS_pronouns.tex", _table(
        body + note, "Perceived gender: pronouns in stated reasons versus the "
        "name-signaled gender of the chosen physician.", "S_pronouns"))


def tab_persona_index(out: Path) -> None:
    cfg = load_config()
    tr = pd.read_csv(ROOT / cfg["paths"]["tables"] / "tradeoffs.csv")
    labels = {"rating_step_43": "Rating 4.3 (vs 3.9)",
              "rating_step_47": "Rating 4.7 (vs 3.9)",
              "volume_85": "85 reviews (vs 12)", "volume_400": "400 reviews (vs 12)",
              "recency": "Recent review", "feedback_response": "Responds to feedback",
              "network": "Hospital-affiliated", "telehealth": "Telehealth",
              "experience_18": "18 yrs (vs 8)", "experience_28": "28 yrs (vs 8)",
              "female_signal": "Female-signaled name",
              "black_signal": "Black-signaled name",
              "hispanic_signal": "Hispanic-signaled name",
              "east_asian_signal": "East Asian-signaled name",
              "south_asian_signal": "South Asian-signaled name",
              "position_first": "First-listed (vs avg.\\ 2--5)"}
    scopes = [("pooled", "Pooled"), ("persona:new_in_town", "New in town"),
              ("persona:chronic", "Chronic care"), ("persona:budget", "Uninsured")]
    rows = []
    for key, lab in labels.items():
        cells = []
        for s, _ in scopes:
            r = tr[(tr["scope"] == s) & (tr["attribute"] == key)].iloc[0]
            cells.append(f"{r['dollars_equiv'] / 100:.2f} "
                         f"{_ci(r['ci_lo'] / 100, r['ci_hi'] / 100, 2)}")
        rows.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    body = ("\\begin{tabular}{lcccc}\n\\toprule\n"
            "Attribute & " + " & ".join(l for _, l in scopes) + " \\\\\n"
            "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    note = ("\\\\[2pt]\n\\parbox{\\textwidth}{\\footnotesize\\textit{Note.} "
            "Fee-range index (prespecified linearized fee-equivalent $\\div$ "
            "\\$100) with Krinsky--Robb 95\\% intervals, pooled and within "
            "persona; each persona uses its own fee AMCE as denominator, so "
            "indices are larger where the fee penalty is smaller.}")
    _write(out, "tabS_persona_index.tex", _table(
        body + note, "Fee-range indices by patient persona.", "S_persona_index",
        resize=True))


def main() -> None:
    cfg = load_config()
    src = ROOT / cfg["paths"]["tables"] / "revision"
    out = src / "latex"
    out.mkdir(parents=True, exist_ok=True)
    tab_heterogeneity(src, out)
    tab_names(src, out)
    tab_feeindex(src, out)
    tab_per_model_full(src, out)
    tab_het_tests(src, out)
    tab_pronouns(src, out)
    tab_persona_index(out)


if __name__ == "__main__":
    main()
