"""Monte-Carlo precision analysis for the planned design.

Simulates choices from a conditional-logit data-generating process with
plausible effect sizes, then reports for each attribute:
  - implied true AMCE (pp change in P(chosen))
  - empirical SE of the AMCE estimator at the planned N
  - power to reject zero at alpha=.05
  - minimum detectable AMCE at 80% power (= 2.8 x empirical SE)
Demographic (name-signaled) effects are included at small assumed sizes so
the planned N is justified for the fairness estimands, not only the
reputation attributes.

Run:  python -m src.power_sim
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.design import load_config

ROOT = Path(__file__).resolve().parents[1]

# Plausible logit-scale effects (per step / vs absent), pre-data assumptions.
# Demographic effects are set SMALL on purpose: the design must be powered to
# detect (or bound) subtle name-signaled effects, not just headline attributes.
TRUE_BETAS = {
    "rating": 0.45,        # per level step (3.9 -> 4.3 -> 4.7)
    "volume": 0.25,        # per level step
    "recency": 0.20,       # recent vs stale
    "feedback_response": 0.15,
    "affiliation": 0.15,   # hospital network vs independent
    "price": -0.30,        # per level step upward
    "telehealth": 0.20,
    "experience": 0.15,    # per level step (8 -> 18 -> 28 years)
    "female": 0.10,        # name-signaled
    "black": -0.15,        # name-signaled, vs white (other groups 0)
    "position1": 0.30,     # first-listed bonus
}

N_ALT = 5
N_REPS = 300


def simulate_once(rng: np.random.Generator, n_trials: int) -> dict:
    # attribute levels coded 0,1,2 (3-level) or 0,1 (binary), iid uniform;
    # ethnicity coded 0-4 (0 = white reference, 1 = black)
    rating = rng.integers(0, 3, (n_trials, N_ALT))
    volume = rng.integers(0, 3, (n_trials, N_ALT))
    recency = rng.integers(0, 2, (n_trials, N_ALT))
    resp = rng.integers(0, 2, (n_trials, N_ALT))
    network = rng.integers(0, 2, (n_trials, N_ALT))
    price = rng.integers(0, 3, (n_trials, N_ALT))
    tele = rng.integers(0, 2, (n_trials, N_ALT))
    exper = rng.integers(0, 3, (n_trials, N_ALT))
    female = rng.integers(0, 2, (n_trials, N_ALT))
    ethnicity = rng.integers(0, 5, (n_trials, N_ALT))
    black = (ethnicity == 1).astype(float)
    pos1 = np.zeros((n_trials, N_ALT)); pos1[:, 0] = 1

    u = (TRUE_BETAS["rating"] * rating + TRUE_BETAS["volume"] * volume
         + TRUE_BETAS["recency"] * recency + TRUE_BETAS["feedback_response"] * resp
         + TRUE_BETAS["affiliation"] * network + TRUE_BETAS["price"] * price
         + TRUE_BETAS["telehealth"] * tele + TRUE_BETAS["experience"] * exper
         + TRUE_BETAS["female"] * female + TRUE_BETAS["black"] * black
         + TRUE_BETAS["position1"] * pos1
         + rng.gumbel(size=(n_trials, N_ALT)))
    chosen = (u == u.max(axis=1, keepdims=True)).astype(float)

    out = {}
    for name, x, hi, lo in [
        ("rating", rating, 2, 0), ("volume", volume, 2, 0),
        ("recency", recency, 1, 0), ("feedback_response", resp, 1, 0),
        ("affiliation", network, 1, 0), ("price", price, 2, 0),
        ("telehealth", tele, 1, 0), ("experience", exper, 2, 0),
        ("female", female, 1, 0), ("black", black, 1, 0),
    ]:
        m_hi, m_lo = chosen[x == hi].mean(), chosen[x == lo].mean()
        n_hi, n_lo = (x == hi).sum(), (x == lo).sum()
        se = np.sqrt(m_hi * (1 - m_hi) / n_hi + m_lo * (1 - m_lo) / n_lo)
        out[name] = (m_hi - m_lo, se)
    out["position1"] = (
        chosen[:, 0].mean() - chosen[:, 1:].mean(),
        np.sqrt(chosen[:, 0].var() / n_trials
                + chosen[:, 1:].var() / (n_trials * (N_ALT - 1))))
    return out


def main() -> None:
    cfg = load_config()
    n_trials = cfg["arms"]["main"]["n_trials"]
    rng = np.random.default_rng(7)

    keys = list(TRUE_BETAS.keys())
    est = {k: [] for k in keys}
    se = {k: [] for k in keys}
    for _ in range(N_REPS):
        res = simulate_once(rng, n_trials)
        for k in keys:
            est[k].append(res[k][0])
            se[k].append(res[k][1])

    rows = []
    for k in keys:
        amce = float(np.mean(est[k]))
        emp_se = float(np.std(est[k]))           # empirical SE across sims
        power = float(np.mean(np.abs(np.array(est[k]) /
                                     np.array(se[k])) > 1.96))
        rows.append(dict(effect=k, true_amce_pp=round(100 * amce, 2),
                         empirical_se_pp=round(100 * emp_se, 2),
                         power_at_assumed_effect=round(power, 3),
                         mde_80pct_power_pp=round(100 * 2.8 * emp_se, 2)))
    table = pd.DataFrame(rows)

    out = ROOT / cfg["paths"]["tables"]
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "power_mde.csv", index=False)
    print(f"Planned main-arm N = {n_trials} trials x {N_ALT} cards")
    print(table.to_string(index=False))
    print(f"\nSaved to {out / 'power_mde.csv'}")


if __name__ == "__main__":
    main()
