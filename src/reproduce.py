"""Regenerate every analysis table and figure from the raw model responses.

Runs the analysis chain in order on data/raw/*.jsonl and the frozen design:
  power_sim (pre-collection precision analysis) -> prep -> amce -> clogit -> tradeoffs -> robustness -> refusals -> reasons
  -> letterarm -> pilot_frontier -> tables -> figures
  -> revision_r1 -> revision_tables   (reviewer-requested, post hoc)

No model is called. Reason coding uses the keyword dictionary only (the
judge-model coder is opt-in: python -m src.analysis.reasons --judge).

Run:  python -m src.reproduce
      python -m src.reproduce --skip-revision   (prespecified analyses only)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STEPS = ["src.power_sim"] + [
    f"src.analysis.{m}" for m in
    ("prep", "amce", "clogit", "tradeoffs", "robustness", "refusals",
     "reasons", "letterarm", "pilot_frontier", "tables", "figures")]
REVISION_STEPS = ["src.analysis.revision_r1", "src.analysis.revision_tables"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-revision", action="store_true")
    args = ap.parse_args()

    steps = STEPS + ([] if args.skip_revision else REVISION_STEPS)
    for i, step in enumerate(steps, 1):
        print(f"[reproduce] ({i}/{len(steps)}) {step}", flush=True)
        done = subprocess.run([sys.executable, "-m", step], cwd=ROOT)
        if done.returncode != 0:
            raise SystemExit(f"[reproduce] step {step} failed "
                             f"(exit {done.returncode})")
    print("[reproduce] done: results/tables, results/figures, "
          "paper/tables regenerated")


if __name__ == "__main__":
    main()
