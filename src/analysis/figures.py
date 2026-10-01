"""Publication figures (matplotlib, colorblind-safe, 300 dpi PDF + PNG).

F1 AMCE forest plot (pooled + per model)      F4 rating x volume diagnosticity
F2 position-bias AMCEs                        F5 stated vs revealed scatter
F3 persona x attribute interactions           F6 robustness summary

Run:  python -m src.analysis.figures
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from src.design import load_config
from src.analysis.amce import load_choices, BASE_FORMULA, fit_lpm

ROOT = Path(__file__).resolve().parents[1].parent

# Okabe-Ito colorblind-safe palette
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9",
           "#D55E00", "#F0E442", "#000000"]

TERM_LABELS = {
    "C(rating)[T.4.3]": "Rating 4.3 (vs 3.9)",
    "C(rating)[T.4.7]": "Rating 4.7 (vs 3.9)",
    "C(volume)[T.85]": "85 reviews (vs 12)",
    "C(volume)[T.400]": "400 reviews (vs 12)",
    "recent": "Recent review (vs 11 mo.)",
    "responds": "Responds to patient feedback",
    "network": "Hospital-affiliated (vs independent)",
    "C(price)[T.140]": r"\$140 fee (vs \$90)",
    "C(price)[T.190]": r"\$190 fee (vs \$90)",
    "telehealth": "Telehealth available",
    "C(experience)[T.18]": "18 yrs in practice (vs 8)",
    "C(experience)[T.28]": "28 yrs in practice (vs 8)",
    "female": "Female-signaled name (vs male)",
    "eth_black": "Black-signaled name (vs White)",
    "eth_hispanic": "Hispanic-signaled name (vs White)",
    "eth_east_asian": "East Asian-signaled name (vs White)",
    "eth_south_asian": "South Asian-signaled name (vs White)",
}
SLOT_LABELS = {f"C(slot)[T.{s}]": f"Position {s} (vs 1)" for s in range(2, 6)}

plt.rcParams.update({
    # typography
    "font.size": 9, "axes.titlesize": 10.5, "axes.labelsize": 9.5,
    "axes.titleweight": "bold", "axes.labelcolor": "#222222",
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8, "font.family": "DejaVu Sans",
    # resolution
    "figure.dpi": 120, "savefig.dpi": 600, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    # spines: keep only left+bottom, lightened
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#666666", "axes.linewidth": 0.8,
    "xtick.color": "#666666", "ytick.color": "#666666",
    "xtick.labelcolor": "#222222", "ytick.labelcolor": "#222222",
    # grid applied per-figure on the value axis only (see _grid)
    "axes.axisbelow": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.7,
})


def _grid(ax, axis: str = "x") -> None:
    """Light grid on the quantitative axis only; keeps categorical axes clean."""
    ax.grid(True, axis=axis, color="#E6E6E6", linewidth=0.7)
    ax.set_axisbelow(True)


def save(fig, name: str) -> None:
    cfg = load_config()
    out = ROOT / cfg["paths"]["figures"]
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] {name} saved")


def f1_forest() -> None:
    cfg = load_config()
    amce = pd.read_csv(ROOT / cfg["paths"]["tables"] / "amce_main.csv")
    amce = amce[amce["term"].isin(TERM_LABELS)]
    scopes = ["pooled"] + sorted(s for s in amce["scope"].unique()
                                 if s != "pooled")
    terms = list(TERM_LABELS)[::-1]

    fig, ax = plt.subplots(figsize=(7, 6.6))  # 17 terms need the taller canvas
    for j, scope in enumerate(scopes):
        sub = amce[amce["scope"] == scope].set_index("term")
        y = np.array([terms.index(t) for t in sub.index if t in terms],
                     dtype=float)
        off = (j - (len(scopes) - 1) / 2) * 0.14
        vals = sub.loc[[terms[int(i)] for i in y]]
        label = scope.replace("model:", "")
        ax.errorbar(vals["amce_pp"], y + off,
                    xerr=[vals["amce_pp"] - vals["ci_lo_pp"],
                          vals["ci_hi_pp"] - vals["amce_pp"]],
                    fmt="o", ms=4 if scope != "pooled" else 6,
                    lw=1.4 if scope == "pooled" else 0.9,
                    color=PALETTE[j % len(PALETTE)],
                    label=label, capsize=0,
                    alpha=1.0 if scope == "pooled" else 0.75)
    _grid(ax, "x")
    ax.axvline(0, color="#444444", lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.set_yticks(range(len(terms)))
    ax.set_yticklabels([TERM_LABELS[t] for t in terms])
    ax.set_xlabel("AMCE: change in P(recommended), percentage points")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right", ncol=2,
              columnspacing=1.0, handletextpad=0.4)
    ax.set_title("Causal effect of physician signals on LLM recommendation")
    ax.margins(y=0.02)
    save(fig, "F1_amce_forest")


def f2_position() -> None:
    cfg = load_config()
    amce = pd.read_csv(ROOT / cfg["paths"]["tables"] / "amce_main.csv")
    pos = amce[(amce["scope"] == "pooled") & (amce["term"].isin(SLOT_LABELS))]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    y = np.arange(len(pos))
    _grid(ax, "x")
    ax.barh(y, pos["amce_pp"], color=PALETTE[0], height=0.62, zorder=2)
    for i, (_, r) in enumerate(pos.iterrows()):
        ax.plot([r["ci_lo_pp"], r["ci_hi_pp"]], [i, i], color="#222222",
                lw=1.1, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([SLOT_LABELS[t] for t in pos["term"]])
    ax.axvline(0, color="#444444", lw=0.9, zorder=1)
    ax.set_xlabel("AMCE vs first-listed (pp)")
    ax.set_title("Position bias in LLM physician recommendation")
    save(fig, "F2_position_bias")


def f3_persona() -> None:
    df = load_choices("main")
    rows = []
    for persona, grp in df.groupby("persona"):
        res = fit_lpm(grp, BASE_FORMULA + " + C(model)")
        for term in ["C(price)[T.190]", "telehealth", "C(experience)[T.28]"]:
            rows.append(dict(persona=persona, term=TERM_LABELS[term],
                             amce=100 * res.params[term],
                             se=100 * res.bse[term]))
    d = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    terms = d["term"].unique()
    personas = sorted(d["persona"].unique())
    x = np.arange(len(terms))
    _grid(ax, "y")
    for j, p in enumerate(personas):
        sub = d[d["persona"] == p].set_index("term").loc[terms]
        ax.bar(x + (j - 1) * 0.25, sub["amce"], width=0.22,
               yerr=1.96 * sub["se"], color=PALETTE[j], label=p,
               error_kw=dict(lw=0.9), zorder=2)
    ax.axhline(0, color="#444444", lw=0.9, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(terms, fontsize=8)
    ax.set_ylabel("AMCE (pp)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Persona-contingent signal weights")
    save(fig, "F3_persona_interactions")


def f4_diagnosticity() -> None:
    df = load_choices("main")
    rows = []
    for vol, grp in df.groupby("volume"):
        for rat, g2 in grp.groupby("rating"):
            rows.append(dict(volume=vol, rating=rat,
                             p=g2["chosen"].mean(),
                             se=g2["chosen"].std() / np.sqrt(len(g2))))
    d = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    _grid(ax, "y")
    for j, vol in enumerate(sorted(d["volume"].unique())):
        sub = d[d["volume"] == vol].sort_values("rating")
        ax.errorbar(sub["rating"], 100 * sub["p"], yerr=196 * sub["se"],
                    marker="o", ms=5, color=PALETTE[j], label=f"{vol} reviews",
                    lw=1.8, capsize=2, zorder=2)
    ax.set_xlabel("Displayed average rating")
    ax.set_ylabel("P(recommended), %")
    ax.legend(frameon=False, fontsize=8, title="Review volume",
              title_fontsize=8)
    ax.set_title("Signal diagnosticity: rating effect by review volume")
    save(fig, "F4_rating_x_volume")


def f5_stated_revealed() -> None:
    cfg = load_config()
    path = ROOT / cfg["paths"]["tables"] / "stated_vs_revealed.csv"
    if not path.exists():
        print("[figures] F5 skipped (stated_vs_revealed.csv missing)")
        return
    d = pd.read_csv(path).dropna(subset=["amce_share"])
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    _grid(ax, "both")
    for j, (model, grp) in enumerate(d.groupby("model")):
        ax.scatter(grp["amce_share"], grp["mention_share"], s=30,
                   color=PALETTE[j % len(PALETTE)], label=model, alpha=0.85,
                   edgecolors="white", linewidths=0.4, zorder=3)
    lim = max(d["amce_share"].max(), d["mention_share"].max()) * 1.1
    ax.plot([0, lim], [0, lim], color="#888888", lw=0.9, ls=(0, (4, 3)),
            zorder=1, label="stated = revealed")
    ax.set_xlabel("Revealed importance (AMCE share)")
    ax.set_ylabel("Stated importance (mention share)")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.set_title("Stated vs revealed attribute importance")
    save(fig, "F5_stated_vs_revealed")


def f7_intersectional() -> None:
    cfg = load_config()
    path = ROOT / cfg["paths"]["tables"] / "intersectional.csv"
    if not path.exists():
        print("[figures] F7 skipped (intersectional.csv missing)")
        return
    d = pd.read_csv(path)
    d = d[d["cell"] != "male_white"]  # reference (AMCE 0 by construction)
    cell_labels = {
        "male_black": "Black male", "male_hispanic": "Hispanic male",
        "male_east_asian": "East Asian male",
        "male_south_asian": "South Asian male", "female_white": "White female",
        "female_black": "Black female", "female_hispanic": "Hispanic female",
        "female_east_asian": "East Asian female",
        "female_south_asian": "South Asian female",
    }
    d = d[d["cell"].isin(cell_labels)]
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    _grid(ax, "x")
    colors = [PALETTE[2] if c.startswith("female") else PALETTE[0]
              for c in d["cell"]]
    ax.barh(y, d["amce_pp"], color=colors, height=0.66, zorder=2)
    for i, (_, r) in enumerate(d.iterrows()):
        ax.plot([r["ci_lo_pp"], r["ci_hi_pp"]], [i, i], color="#222222",
                lw=1.0, zorder=3)
    ax.axvline(0, color="#444444", lw=0.9, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([cell_labels[c] for c in d["cell"]])
    ax.set_xlabel("AMCE vs White-male-signaled names (pp)")
    ax.set_title("Name-signaled gender $\\times$ ethnicity cells")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=PALETTE[0], label="Male-signaled"),
                       Patch(color=PALETTE[2], label="Female-signaled")],
              frameon=False, fontsize=8, loc="lower right")
    save(fig, "F7_intersectional")


def f6_robustness() -> None:
    cfg = load_config()
    tdir = ROOT / cfg["paths"]["tables"]
    panels = []
    for fname, label in [("robustness_temperature.csv", "Temperature"),
                         ("robustness_grounded.csv", "Snippet format"),
                         ("robustness_field_order.csv", "Field order")]:
        p = tdir / fname
        if not p.exists():
            continue
        d = pd.read_csv(p)
        # The temperature table is long-format (one row per condition x term);
        # reshape it to a diff_pp column (AMCE at 0.7 minus AMCE at 0.0) so it
        # renders on the same footing as the other robustness panels.
        if "condition" in d.columns and "diff_pp" not in d.columns:
            w = d.pivot_table(index="term", columns="condition",
                              values="amce_pp")
            hi = "temp_0.7" if "temp_0.7" in w.columns else w.columns[-1]
            lo = "temp_0.0" if "temp_0.0" in w.columns else w.columns[0]
            d = (w[hi] - w[lo]).reset_index()
            d.columns = ["term", "diff_pp"]
        panels.append((label, d))
    if not panels:
        print("[figures] F6 skipped (no robustness tables yet)")
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.4),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (label, d) in zip(axes, panels):
        if "diff_pp" in d.columns:
            _grid(ax, "x")
            ax.barh(np.arange(len(d)), d["diff_pp"], color=PALETTE[0],
                    height=0.62, zorder=2)
            ax.set_yticks(np.arange(len(d)))
            ax.set_yticklabels([TERM_LABELS.get(t, t) for t in d["term"]],
                               fontsize=7)
            ax.axvline(0, color="#444444", lw=0.9, zorder=1)
            ax.set_xlabel("AMCE difference (pp)")
        ax.set_title(label, fontsize=9.5)
    fig.suptitle("Robustness of signal weights", y=1.02, fontweight="bold")
    save(fig, "F6_robustness")


def main() -> None:
    f1_forest()
    f2_position()
    f3_persona()
    f4_diagnosticity()
    f5_stated_revealed()
    f6_robustness()
    f7_intersectional()


if __name__ == "__main__":
    main()
