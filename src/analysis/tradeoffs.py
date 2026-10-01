"""Fee-equivalent trade-offs with Krinsky-Robb CIs.

Preregistered gate: fee AMCEs must be monotone decreasing across
$90/$140/$190 before any trade-off is computed.

"Fee-equivalent" = AMCE of an attribute divided by the per-dollar fee
AMCE (linearized over the $100 range). Reported pooled and per persona;
headline includes the position-1 advantage and the name-signaled demographic
effects in $/visit.

Output: results/tables/tradeoffs.csv

Run:  python -m src.analysis.tradeoffs
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from src.design import load_config
from src.analysis.amce import load_choices, BASE_FORMULA

ROOT = Path(__file__).resolve().parents[1].parent
N_DRAWS = 10_000

TERMS = {
    "rating_step_43": "C(rating)[T.4.3]",
    "rating_step_47": "C(rating)[T.4.7]",
    "volume_85": "C(volume)[T.85]",
    "volume_400": "C(volume)[T.400]",
    "recency": "recent",
    "feedback_response": "responds",
    "network": "network",
    "telehealth": "telehealth",
    "experience_18": "C(experience)[T.18]",
    "experience_28": "C(experience)[T.28]",
    "female_signal": "female",
    "black_signal": "eth_black",
    "hispanic_signal": "eth_hispanic",
    "east_asian_signal": "eth_east_asian",
    "south_asian_signal": "eth_south_asian",
    "position_first": None,  # handled via slot dummies (avg of slots 2-5)
}


def monotone_price(res) -> bool:
    p140 = res.params.get("C(price)[T.140]", np.nan)
    p190 = res.params.get("C(price)[T.190]", np.nan)
    return (p140 < 0) and (p190 < p140)


def krinsky_robb(res, df_scope: str, rng: np.random.Generator) -> list[dict]:
    params = res.params
    draws = rng.multivariate_normal(params.to_numpy(), res.cov_params().to_numpy(),
                                    size=N_DRAWS)
    cols = list(params.index)

    def col(name: str) -> np.ndarray:
        return draws[:, cols.index(name)]

    per_dollar = -col("C(price)[T.190]") / 100.0  # positive = penalty per $
    slot_cols = [c for c in cols if c.startswith("C(slot)")]
    pos1_adv = -np.mean([col(c) for c in slot_cols], axis=0)

    rows = []
    for label, term in TERMS.items():
        numer = pos1_adv if term is None else col(term)
        ratio = numer / per_dollar
        est = float(np.median(ratio))
        lo, hi = np.percentile(ratio, [2.5, 97.5])
        rows.append(dict(scope=df_scope, attribute=label,
                         dollars_equiv=round(est, 1),
                         ci_lo=round(float(lo), 1), ci_hi=round(float(hi), 1)))
    return rows


def main() -> None:
    cfg = load_config()
    df = load_choices("main")
    rng = np.random.default_rng(11)
    out_rows = []

    scopes = [("pooled", df)] + [(f"persona:{p}", g)
                                 for p, g in df.groupby("persona")]
    for scope, d in scopes:
        formula = BASE_FORMULA + (" + C(model)" if d["model"].nunique() > 1 else "")
        res = smf.ols(formula, data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d["trial_id"]})
        if not monotone_price(res):
            out_rows.append(dict(scope=scope, attribute="GATE_FAILED",
                                 dollars_equiv=np.nan, ci_lo=np.nan,
                                 ci_hi=np.nan))
            print(f"[tradeoffs] {scope}: price monotonicity gate FAILED - "
                  f"no trade-offs computed (per preregistration)")
            continue
        out_rows.extend(krinsky_robb(res, scope, rng))

    out = ROOT / cfg["paths"]["tables"]
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(out / "tradeoffs.csv", index=False)
    print(f"[tradeoffs] written ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()
