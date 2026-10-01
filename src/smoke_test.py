"""End-to-end pipeline smoke test against a known data-generating process.

Runs the FULL experiment pipeline (runner -> prep -> amce -> clogit ->
tradeoffs -> reasons -> robustness -> figures [-> letterarm/tables if
present]) with mock LLM clients whose choice behavior follows known
conditional-logit effects (src/mock_client.py), inside a sandboxed config
(config/experiment_smoke.yaml). Then asserts the pipeline RECOVERS the known
effects. This validates every line of pipeline code without any GPU use.

Run:  python -m src.smoke_test            (~2-5 min, CPU only)
      python -m src.smoke_test --keep     (keep sandbox outputs for inspection)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
ENV = {**os.environ, "AUDIT_CONFIG": str(ROOT / "config" / "experiment_smoke.yaml"),
       "MPLBACKEND": "Agg"}

MODELS = ["mock-alpha", "mock-beta"]
ARMS = ("main,temp0,grounded,rank_top3,field_order,retest,letter,"
        "nopersona,card_reversed,manip")

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def run(args: list[str], label: str) -> None:
    t0 = time.time()
    r = subprocess.run([PY, "-m", *args], cwd=ROOT, env=ENV,
                       capture_output=True, text=True)
    dt = time.time() - t0
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-3000:])
        raise SystemExit(f"step failed: {label} (exit {r.returncode})")
    print(f"[smoke] {label} OK ({dt:.0f}s)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    for d in [ROOT / "data" / "smoke", ROOT / "results" / "smoke"]:
        if d.exists():
            shutil.rmtree(d)

    print("[smoke] phase 1: mock experiment runs")
    for m in MODELS:
        run(["src.runner", "--model", m, "--arms", ARMS], f"runner {m}")

    print("[smoke] phase 2: analysis pipeline")
    run(["src.analysis.prep"], "prep")
    run(["src.analysis.amce"], "amce main")
    run(["src.analysis.amce", "--arm", "temp0"], "amce temp0")
    run(["src.analysis.clogit"], "clogit")
    run(["src.analysis.tradeoffs"], "tradeoffs")
    run(["src.analysis.reasons"], "reasons (dictionary)")
    run(["src.analysis.refusals"], "refusals")
    run(["src.analysis.robustness"], "robustness")
    run(["src.analysis.figures"], "figures")
    if (ROOT / "src" / "analysis" / "letterarm.py").exists():
        run(["src.analysis.letterarm"], "letterarm")
    if (ROOT / "src" / "analysis" / "tables.py").exists():
        run(["src.analysis.tables"], "tables")

    print("[smoke] phase 3: recovery assertions")
    tdir = ROOT / "results" / "smoke" / "tables"

    fails = pd.read_csv(tdir / "parse_failures.csv")
    check("parse failure rate < 3% everywhere",
          (fails["fail_rate"] < 0.03).all(),
          f"max={fails['fail_rate'].max():.3f}")

    amce = pd.read_csv(tdir / "amce_main.csv")
    pooled = amce[amce["scope"] == "pooled"].set_index("term")

    sign_terms = {"C(rating)[T.4.7]": 1, "C(volume)[T.400]": 1, "recent": 1,
                  "responds": 1, "network": 1, "C(price)[T.190]": -1,
                  "telehealth": 1, "C(experience)[T.28]": 1,
                  "female": 1, "eth_black": -1}
    ok_signs = all((pooled.loc[t, "amce_pp"] * s > 0) and
                   (pooled.loc[t, "p"] < 0.05)
                   for t, s in sign_terms.items())
    check("all 10 attribute AMCEs (incl. female +, black -): sign & p<.05",
          ok_signs)

    null_eths = ["eth_hispanic", "eth_east_asian", "eth_south_asian"]
    ok_nulls = all(abs(pooled.loc[t, "amce_pp"]) < 1.5 for t in null_eths)
    check("true-null ethnicity AMCEs stay within +/-1.5 pp", ok_nulls,
          ", ".join(f"{t}={pooled.loc[t, 'amce_pp']:.2f}" for t in null_eths))

    r47 = pooled.loc["C(rating)[T.4.7]", "amce_pp"]
    check("rating 4.7 AMCE in [12, 26] pp", 12 <= r47 <= 26, f"{r47:.1f}")
    p190 = pooled.loc["C(price)[T.190]", "amce_pp"]
    check("fee $190 AMCE in [-16, -3] pp", -16 <= p190 <= -3, f"{p190:.1f}")

    slots = [pooled.loc[f"C(slot)[T.{s}]", "amce_pp"] for s in range(2, 6)]
    check("position-1 advantage recovered (all slot dummies negative)",
          all(v < 0 for v in slots), f"mean={sum(slots)/4:.1f}")

    conf = pd.read_csv(tdir / "confirmatory_tests.csv").set_index("hypothesis")
    h110 = [f"H{i}" for i in range(1, 11)]
    ok_h = all(conf.loc[[h for h in conf.index if h.startswith(k + "_")],
                        "p_holm"].iloc[0] < 0.05 for k in h110)
    check("H1-H10 all Holm-significant (incl. gender & ethnicity)", ok_h)
    # H12 (rating x volume) is NOT asserted either way: a logit DGP with no
    # utility-scale interaction still yields probability-scale interactions
    # in an LPM (same reason the hotel audit's smoke asserted neither H8/H9).
    # H11 (persona x price) and H13 (gender x ethnicity) ARE null in the mock
    # DGP (purely additive), so both must stay non-significant.
    null_int = [k for k in conf.index if k.startswith(("H11", "H13"))]
    ok_null = (conf.loc[null_int, "p_holm"] > 0.05).all()
    check("H11 & H13 null (additive DGP: no persona/intersection effects)",
          ok_null, "no false positives")

    clog = pd.read_csv(tdir / "clogit_main.csv")
    cp = clog[clog["scope"] == "pooled"].set_index("term")
    b47, bp190 = cp.loc["rating_47", "beta"], cp.loc["price_190", "beta"]
    check("clogit beta(rating 4.7) ~ truth 0.90-1.06 [0.75, 1.25]",
          0.75 <= b47 <= 1.25, f"{b47:.3f}")
    check("clogit beta(fee 190) ~ truth -0.60 [-0.80, -0.42]",
          -0.80 <= bp190 <= -0.42, f"{bp190:.3f}")
    bblk = cp.loc["eth_black", "beta"]
    check("clogit beta(black name) ~ truth -0.15 [-0.30, -0.03]",
          -0.30 <= bblk <= -0.03, f"{bblk:.3f}")

    trade = pd.read_csv(tdir / "tradeoffs.csv")
    pooled_t = trade[trade["scope"] == "pooled"]
    gate_ok = not (pooled_t["attribute"] == "GATE_FAILED").any()
    check("price monotonicity gate passes", gate_ok)
    if gate_ok:
        pt = pooled_t.set_index("attribute")
        r47d = pt.loc["rating_step_47", "dollars_equiv"]
        check("rating-4.7 fee-equivalent positive & finite",
              0 < r47d < 1000, f"${r47d:.0f}")
        blkd = pt.loc["black_signal", "dollars_equiv"]
        check("black-name fee-equivalent negative (penalty recovered)",
              -1000 < blkd < 0, f"${blkd:.0f}")

    plac = pd.read_csv(tdir / "robustness_name_placebo.csv").iloc[0]
    check("name placebo null (p > .005)", plac["p_value"] > 0.005,
          f"p={plac['p_value']:.3f}")

    retest = pd.read_csv(tdir / "robustness_retest_summary.csv")
    check("retest consistency sane (0.2 <= modal share <= 1)",
          retest["modal_share"].between(0.2, 1.0).all())

    svr = pd.read_csv(tdir / "stated_vs_revealed.csv")
    check("stated-vs-revealed produced with mention shares",
          (svr["mention_share"] > 0).any())

    # Intersectional cells: additive DGP => female_black ~ female + black.
    inter = pd.read_csv(tdir / "intersectional.csv").set_index("cell")
    fw = inter.loc["female_white", "amce_pp"]
    mb = inter.loc["male_black", "amce_pp"]
    fb = inter.loc["female_black", "amce_pp"]
    check("intersectional female_white AMCE positive", fw > 0, f"{fw:.1f}")
    check("intersectional male_black AMCE negative", mb < 0, f"{mb:.1f}")
    check("female_black ~ additive(female_white + male_black) within 3 pp",
          abs(fb - (fw + mb)) < 3.0, f"fb={fb:.1f} vs {fw + mb:.1f}")

    # Refusals: mock injects ~1.5% 'any of these...' abstentions, never any
    # demographic-aware language -> abstain small, demo-aware ~0.
    refu = pd.read_csv(tdir / "refusals_summary.csv")
    rmain = refu[refu["arm"] == "main"]
    check("abstention detected but rare in main arm (0 < rate < 5%)",
          (rmain["abstain_rate"] > 0).any()
          and (rmain["abstain_rate"] < 0.05).all(),
          f"max={rmain['abstain_rate'].max():.3f}")
    check("demographic-aware rate ~0 (mock never cites names) < 0.5%",
          (rmain["demo_aware_rate"] < 0.005).all(),
          f"max={rmain['demo_aware_rate'].max():.4f}")

    fdir = ROOT / "results" / "smoke" / "figures"
    figs = [f"F{i}" for i in (1, 2, 3, 4, 5)]
    missing = [f for f in figs
               if not list(fdir.glob(f"{f}_*.pdf"))]
    check("figures F1-F5 rendered", not missing, f"missing={missing or 'none'}")

    n_fail = sum(1 for _, ok, _ in CHECKS if not ok)
    print(f"\n[smoke] {len(CHECKS) - n_fail}/{len(CHECKS)} checks passed")
    if not args.keep and n_fail == 0:
        shutil.rmtree(ROOT / "data" / "smoke", ignore_errors=True)
        shutil.rmtree(ROOT / "results" / "smoke", ignore_errors=True)
        print("[smoke] sandbox cleaned")
    if n_fail:
        raise SystemExit(1)
    print("[smoke] PIPELINE VERIFIED END-TO-END")


if __name__ == "__main__":
    main()
