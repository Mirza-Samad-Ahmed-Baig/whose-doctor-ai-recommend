"""Render results/tables CSVs into publication LaTeX tables (booktabs).

Each table is written as a self-contained file under paper/tables/ wrapped in
a table float with caption + label. Missing input CSVs are skipped with a
console note (the analysis pipeline produces them incrementally), so this can
be run at any point.

Run:  python -m src.analysis.tables
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.design import load_config
from src.analysis.figures import TERM_LABELS, SLOT_LABELS

ROOT = Path(__file__).resolve().parents[1].parent

# Canonical attribute order for the main AMCE table (matches figures.py).
AMCE_TERM_ORDER = list(TERM_LABELS) + list(SLOT_LABELS)
ALL_LABELS = {**TERM_LABELS, **SLOT_LABELS}

# Human labels for tradeoff attribute keys (src/analysis/tradeoffs.py TERMS).
TRADEOFF_LABELS = {
    "rating_step_43": "Rating 4.3 (vs 3.9)",
    "rating_step_47": "Rating 4.7 (vs 3.9)",
    "volume_85": "85 reviews (vs 12)",
    "volume_400": "400 reviews (vs 12)",
    "recency": "Recent review (vs 11 mo.)",
    "feedback_response": "Responds to patient feedback",
    "network": "Hospital-affiliated (vs independent)",
    "telehealth": "Telehealth available",
    "experience_18": "18 yrs in practice (vs 8)",
    "experience_28": "28 yrs in practice (vs 8)",
    "female_signal": "Female name (vs male)",
    "black_signal": "Black name (vs White)",
    "hispanic_signal": "Hispanic name (vs White)",
    "east_asian_signal": "East Asian name (vs White)",
    "south_asian_signal": "South Asian name (vs White)",
    "position_first": "First-listed (vs avg 2-5)",
}
TRADEOFF_SCOPES = ["pooled", "persona:new_in_town", "persona:chronic",
                   "persona:budget"]
SCOPE_HEADERS = {"pooled": "Pooled", "persona:new_in_town": "New in town",
                 "persona:chronic": "Chronic care", "persona:budget": "Budget"}

# Human labels for importance_shares attribute keys (amce.py importance_shares).
IMPORTANCE_LABELS = {
    "rating": "Patient rating", "volume": "Review volume", "price": "Visit fee",
    "recency": "Recency", "response": "Feedback response",
    "affiliation": "Affiliation", "telehealth": "Telehealth",
    "experience": "Experience", "gender": "Gender (name-signaled)",
    "ethnicity": "Ethnicity (name-signaled)", "position": "List position",
}


def _decomp_label(term: str) -> str:
    """Human label for a letterarm coefficient term.

    Slot terms reuse SLOT_LABELS ("Position 2 (vs 1)"); letter terms render as
    "Letter B (vs A)"; anything else falls back to the raw term.
    """
    if term in ALL_LABELS:
        return ALL_LABELS[term]
    m = re.fullmatch(r"C\(letter\)\[T\.([A-E])\]", term)
    if m:
        return f"Letter {m.group(1)} (vs A)"
    return term


def _esc(s) -> str:
    """Escape LaTeX-special characters in free text / model ids."""
    out = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}")]:
        out = out.replace(a, b)
    return out


def _fmt_p(p: float) -> str:
    if pd.isna(p):
        return "--"
    p = float(p)
    if p < 0.001:
        return "<.001"
    s = f"{p:.3f}"
    return s.lstrip("0") if 0 < p < 1 else s  # ".045" style, leading 0 dropped


def _stars(p: float) -> str:
    if pd.isna(p):
        return ""
    p = float(p)
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def _table(body: str, caption: str, label: str) -> str:
    return ("\\begin{table}[t]\n\\centering\n"
            f"\\caption{{{caption}}}\n\\label{{tab:{label}}}\n"
            f"{body}\n\\end{{table}}\n")


def _resize(tabular: str) -> str:
    """Scale a wide tabular down to the text width (keeps it inside the margins)."""
    return "\\resizebox{\\textwidth}{!}{%\n" + tabular + "\n}"


def _read(tdir: Path, name: str) -> pd.DataFrame | None:
    p = tdir / name
    if not p.exists():
        print(f"[tables] skip {name} (not found)")
        return None
    return pd.read_csv(p)


# --------------------------------------------------------------------------
# Individual tables
# --------------------------------------------------------------------------

def tab_amce_main(tdir: Path, out: Path) -> None:
    df = _read(tdir, "amce_main.csv")
    if df is None:
        return
    df = df[df["scope"] == "pooled"].set_index("term")
    rows = []
    for term in AMCE_TERM_ORDER:
        if term not in df.index:
            continue
        r = df.loc[term]
        ci = f"[{r['ci_lo_pp']:.2f}, {r['ci_hi_pp']:.2f}]"
        # labels are already LaTeX (e.g. r"\$140"); escaping them again
        # printed a literal "\{}" before every dollar sign
        rows.append(f"{ALL_LABELS[term]} & {r['amce_pp']:.2f} & {ci} "
                    f"& {_fmt_p(r['p'])} \\\\")
    body = ("\\begin{tabular}{lrrr}\n\\toprule\n"
            "Attribute & AMCE (pp) & 95\\% CI & $P$ \\\\\n\\midrule\n"
            + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}")
    (out / "tab_amce_main.tex").write_text(
        _table(body,
               "Average marginal component effects on P(recommended), pooled "
               "across the seven-model panel (percentage points). Pooled "
               "estimates are panel averages; per-model estimates are in "
               "Table~\\ref{tab:heterogeneity} and Supplementary Table~S2.",
               "amce_main"), encoding="utf-8")
    print("[tables] wrote tab_amce_main.tex")


def tab_confirmatory(tdir: Path, out: Path) -> None:
    df = _read(tdir, "confirmatory_tests.csv")
    if df is None:
        return
    rows = []
    for _, r in df.iterrows():
        name = {"H9_gender": "H9_gender (name-signaled)",
                "H10_ethnicity": "H10_ethnicity (name-signaled)"}.get(
                    r["hypothesis"], r["hypothesis"])
        rows.append(f"{_esc(name)} & {r['wald']:.2f} & {int(r['df'])} "
                    f"& {_fmt_p(r['p_raw'])} & {_fmt_p(r['p_holm'])}"
                    f"{_stars(r['p_holm'])} \\\\")
    body = ("\\begin{tabular}{lrrrr}\n\\toprule\n"
            "Hypothesis & Wald & df & $P$ (raw) & $P$ (Holm) \\\\\n\\midrule\n"
            + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n"
            "\\\\[2pt]\n{\\footnotesize\\textit{Note.} Holm-corrected over all "
            "thirteen confirmatory tests (H1--H13), as "
            "implemented in the analysis code; the submitted version of this note "
            "incorrectly read ``H1--H12''. Tested alone, H13 has raw $P=.0495$. "
            "$^{*}p<.05$, $^{**}p<.01$, $^{***}p<.001$.}")
    (out / "tab_confirmatory.tex").write_text(
        _table(body, "Confirmatory hypothesis tests (joint Wald tests, "
                     "Holm-corrected).", "confirmatory"), encoding="utf-8")
    print("[tables] wrote tab_confirmatory.tex")


def tab_importance(tdir: Path, out: Path) -> None:
    df = _read(tdir, "importance_shares.csv")
    if df is None:
        return
    df = df.sort_values("range_pp", ascending=False)
    rows = [f"{_esc(IMPORTANCE_LABELS.get(r['attribute'], r['attribute']))} "
            f"& {r['range_pp']:.2f} & {r['share']:.3f} \\\\"
            for _, r in df.iterrows()]
    body = ("\\begin{tabular}{lrr}\n\\toprule\n"
            "Attribute & Range (pp) & Share \\\\\n\\midrule\n"
            + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}")
    (out / "tab_importance.tex").write_text(
        _table(body, "Attribute importance: range of pooled AMCEs and normalized "
                     "share (panel average; shares vary by model, "
                     "Supplementary Table~S2).", "importance"), encoding="utf-8")
    print("[tables] wrote tab_importance.tex")


def tab_tradeoffs(tdir: Path, out: Path) -> None:
    df = _read(tdir, "tradeoffs.csv")
    if df is None:
        return
    scopes = [s for s in TRADEOFF_SCOPES if s in set(df["scope"])]
    failed = {s for s in scopes
              if (df[(df["scope"] == s)]["attribute"] == "GATE_FAILED").any()}
    # attribute rows in canonical order, drop the sentinel
    attrs = [a for a in TRADEOFF_LABELS
             if a in set(df.loc[df["attribute"] != "GATE_FAILED", "attribute"])]

    def cell(scope: str, attr: str, first_row: bool) -> str:
        if scope in failed:
            # whole column reads "gate failed (monotonicity)", spanned once
            return (f"\\multirow{{{len(attrs)}}}{{*}}{{gate failed (monotonicity)}}"
                    if first_row else "")
        sub = df[(df["scope"] == scope) & (df["attribute"] == attr)]
        if sub.empty:
            return "--"
        r = sub.iloc[0]
        return f"{r['dollars_equiv']:.1f} [{r['ci_lo']:.1f}, {r['ci_hi']:.1f}]"

    header = "Attribute & " + " & ".join(SCOPE_HEADERS[s] for s in scopes) + " \\\\"
    body_rows = []
    for i, attr in enumerate(attrs):
        cells = [cell(s, attr, first_row=(i == 0)) for s in scopes]
        body_rows.append(f"{_esc(TRADEOFF_LABELS[attr])} & "
                         + " & ".join(cells) + " \\\\")
    colspec = "l" + "c" * len(scopes)
    body = _resize("\\begin{tabular}{" + colspec + "}\n\\toprule\n"
            + header + "\n\\midrule\n"
            + "\n".join(body_rows)
            + "\n\\bottomrule\n\\end{tabular}")
    (out / "tab_tradeoffs.tex").write_text(
        _table(body, "Fee-equivalent trade-offs (\\$/visit) with "
                     "Krinsky--Robb 95\\% CIs.", "tradeoffs"), encoding="utf-8")
    print("[tables] wrote tab_tradeoffs.tex")


def tab_parse(tdir: Path, out: Path) -> None:
    df = _read(tdir, "parse_failures.csv")
    if df is None:
        return
    pivot = df.pivot_table(index="model", columns="arm", values="fail_rate")
    arms = list(pivot.columns)
    header = "Model & " + " & ".join(_esc(a) for a in arms) + " \\\\"
    rows = []
    for model, r in pivot.iterrows():
        cells = ["--" if pd.isna(v) else f"{100 * v:.1f}" for v in r]
        rows.append(f"{_esc(model)} & " + " & ".join(cells) + " \\\\")
    colspec = "l" + "r" * len(arms)
    body = (_resize("\\begin{tabular}{" + colspec + "}\n\\toprule\n"
            + header + "\n\\midrule\n" + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}")
            + "\n\\\\[2pt]\n{\\footnotesize\\textit{Note.} Parse-failure rate (\\%) "
            "by model and arm; pilot and manipulation-check calls excluded.}")
    (out / "tab_parse.tex").write_text(
        _table(body, "Response parse-failure rates.", "parse"),
        encoding="utf-8")
    print("[tables] wrote tab_parse.tex")


def tab_retest(tdir: Path, out: Path) -> None:
    df = _read(tdir, "robustness_retest_summary.csv")
    if df is None:
        return
    rows = [f"{_esc(r['model'])} & {r['modal_share']:.3f} & {r['entropy']:.3f} \\\\"
            for _, r in df.iterrows()]
    body = ("\\begin{tabular}{lrr}\n\\toprule\n"
            "Model & Modal share & Entropy \\\\\n\\midrule\n"
            + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n"
            "\\\\[2pt]\n{\\footnotesize\\textit{Note.} Mean over test--retest "
            "choice sets (5 repetitions each).}")
    (out / "tab_retest.tex").write_text(
        _table(body, "Test--retest consistency of recommendations.", "retest"),
        encoding="utf-8")
    print("[tables] wrote tab_retest.tex")


def tab_letter(tdir: Path, out: Path) -> None:
    df = _read(tdir, "letter_decomposition.csv")
    if df is None:
        return
    caption = ("Letter-arm decomposition of list-position effects from "
               "letter-token preferences (linear probability model on "
               "first-token letter probabilities, trial-clustered SEs).")
    # Fallback: letterarm wrote a single note row (no logprob data captured).
    if "section" not in df.columns or "note" in df.columns:
        note = str(df.iloc[0].get("note", "")) if len(df) else ""
        body = f"{_esc(note)}"
        (out / "tab_letter.tex").write_text(
            _table(body, caption, "letter"), encoding="utf-8")
        print("[tables] wrote tab_letter.tex (note-only: no logprob data)")
        return

    coef = df[df["section"] == "coefficient"]
    rows = [f"{_esc(_decomp_label(r['term']))} & {float(r['coef_pp']):.2f} "
            f"& {float(r['se_pp']):.2f} & {_fmt_p(r['p'])} \\\\"
            for _, r in coef.iterrows()]

    pr = df[df["section"] == "partial_r2"].set_index("term")
    def _added(key: str) -> str:
        return (f"{float(pr.loc[key, 'added_r2']):.4f}"
                if key in pr.index else "--")
    note_line = ("\\\\[2pt]\n{\\footnotesize\\textit{Note.} Position block "
                 f"incremental $R^2$ = {_added('slot | letter')}; "
                 f"letter-token block = {_added('letter | slot')}. "
                 "Slot and letter are independently randomized by design.}")

    body = ("\\begin{tabular}{lrrr}\n\\toprule\n"
            "Term & Coef. (pp) & SE (pp) & $P$ \\\\\n\\midrule\n"
            + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n" + note_line)
    (out / "tab_letter.tex").write_text(
        _table(body, caption, "letter"), encoding="utf-8")
    print("[tables] wrote tab_letter.tex")


CELL_LABELS = {
    "male_white": "White male (ref.)", "male_black": "Black male",
    "male_hispanic": "Hispanic male", "male_east_asian": "East Asian male",
    "male_south_asian": "South Asian male", "female_white": "White female",
    "female_black": "Black female", "female_hispanic": "Hispanic female",
    "female_east_asian": "East Asian female",
    "female_south_asian": "South Asian female",
}


def tab_intersectional(tdir: Path, out: Path) -> None:
    df = _read(tdir, "intersectional.csv")
    if df is None:
        return
    rows = []
    for _, r in df.iterrows():
        if r["cell"] == "male_white":
            rows.append(f"{_esc(CELL_LABELS.get(r['cell'], r['cell']))} "
                        f"& 0.00 & -- & -- & -- \\\\")
            continue
        ci = f"[{r['ci_lo_pp']:.2f}, {r['ci_hi_pp']:.2f}]"
        rows.append(f"{_esc(CELL_LABELS.get(r['cell'], r['cell']))} "
                    f"& {r['amce_pp']:.2f} & {ci} & {_fmt_p(r['p'])} "
                    f"& {_fmt_p(r['tost_p'])} \\\\")
    body = ("\\begin{tabular}{lrrrr}\n\\toprule\n"
            "Gender $\\times$ ethnicity cell & AMCE (pp) & 95\\% CI "
            "& $P$ & TOST $P$ \\\\\n\\midrule\n"
            + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n"
            "\\\\[2pt]\n{\\footnotesize\\textit{Note.} Effect of each "
            "name-signaled cell vs.\\ White male, from a single 10-level "
            "categorical (choice-set--clustered SEs). TOST $p<.05$ indicates "
            "statistical equivalence within $\\pm$1.5 pp.}")
    (out / "tab_intersectional.tex").write_text(
        _table(body, "Name-signaled gender $\\times$ ethnicity cell effects on "
                     "P(recommended).", "intersectional"), encoding="utf-8")
    print("[tables] wrote tab_intersectional.tex")


def tab_refusals(tdir: Path, out: Path) -> None:
    df = _read(tdir, "refusals_summary.csv")
    if df is None or "model" not in df.columns:
        return
    main = df[df["arm"] == "main"]
    if main.empty:
        main = df
    rows = [f"{_esc(r['model'])} & {int(r['n'])} & {100*r['abstain_rate']:.1f} "
            f"& {100*r['demo_aware_rate']:.1f} \\\\"
            for _, r in main.iterrows()]
    body = ("\\begin{tabular}{lrrr}\n\\toprule\n"
            "Model & $n$ & Abstain (\\%) & Demographic-aware (\\%) \\\\\n"
            "\\midrule\n" + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n"
            "\\\\[2pt]\n{\\footnotesize\\textit{Note.} Main arm. "
            "The prompt required a single choice and offered no option to "
            "decline, so abstention measures departures from the instruction. "
            "\\emph{Abstain} = declines to choose or calls the options "
            "equal; \\emph{demographic-aware} = the response text refers to a "
            "physician's name, gender, ethnicity, or race.}")
    (out / "tab_refusals.tex").write_text(
        _table(body, "Abstention and demographic-aware response rates (main "
                     "arm, forced-choice instruction).",
               "refusals"), encoding="utf-8")
    print("[tables] wrote tab_refusals.tex")


def main() -> None:
    cfg = load_config()
    tdir = ROOT / cfg["paths"]["tables"]
    out = ROOT / cfg["paths"].get("paper_tables", "paper/tables")
    out.mkdir(parents=True, exist_ok=True)

    tab_amce_main(tdir, out)
    tab_confirmatory(tdir, out)
    tab_importance(tdir, out)
    tab_intersectional(tdir, out)
    tab_tradeoffs(tdir, out)
    tab_parse(tdir, out)
    tab_retest(tdir, out)
    tab_letter(tdir, out)
    tab_refusals(tdir, out)
    print(f"[tables] done; LaTeX tables under {out}")


if __name__ == "__main__":
    main()
