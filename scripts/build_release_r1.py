"""Build the public replication package for revision R1 (IJMEDI-D-26-04763).

Writes dist/data_code_release_r1/ and dist/data_code_release_r1.zip.
Internal planning notes (docs/positioning.md, docs/titles.md) are excluded.

Run:  python scripts/build_release_r1.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.design import FIRST_NAMES, SURNAMES, load_config  # noqa: E402
from src.stimuli import (TEMPLATES, FORMAT_INSTRUCTIONS, RETRY_REMINDER,  # noqa: E402
                         MANIPULATION_CHECK, NEUTRAL_PERSONA, build_prompt)

OUT = ROOT / "dist" / "data_code_release_r1"
OLLAMA_MANIFESTS = (Path.home() / ".ollama" / "models" / "manifests"
                    / "registry.ollama.ai" / "library")

COPY_DIRS = ["config", "src", "data/design", "data/raw", "data/raw_session_pilot",
             "results/tables", "results/figures"]
COPY_FILES = ["LICENSE", "data/processed/choices.parquet",
              "docs/analysis_plan.md", "docs/COAUTHOR_RUNBOOK.md",
              "scripts/run_azure_gpt4omini.py", "scripts/build_release_r1.py"]


def _copy() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    for d in COPY_DIRS:
        shutil.copytree(ROOT / d, OUT / d,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for f in COPY_FILES:
        (OUT / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, OUT / f)
    req = (ROOT / "dist" / "data_code_release" / "requirements.txt")
    shutil.copy2(req, OUT / "requirements.txt")


def _digests() -> None:
    rows = {}
    for mf in sorted(OLLAMA_MANIFESTS.glob("*/*")):
        tag = f"{mf.parent.name}:{mf.name}"
        layers = json.loads(mf.read_text(encoding="utf-8"))["layers"]
        w = [l for l in layers if l["mediaType"].endswith("image.model")]
        if w:
            rows[tag] = dict(weights_digest=w[0]["digest"], size=w[0]["size"])
    cfg = load_config()
    panel = [m["id"] for m in cfg["models"]["local"]] + [cfg["reason_judge"]["id"]]
    out = {m: rows.get(m, "not recoverable: manifest removed before archiving")
           for m in panel}
    out["gpt-4o-mini"] = ("Azure OpenAI deployment 'gpt-4o-mini', API version "
                          "2025-01-01-preview; model snapshot not logged")
    (OUT / "docs" / "model_digests.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")


def _names() -> None:
    with open(OUT / "docs" / "name_lists.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kind", "gender", "ethnicity", "name"])
        for (g, e), names in FIRST_NAMES.items():
            for n in names:
                w.writerow(["first", g, e, n])
        for e, names in SURNAMES.items():
            for n in names:
                w.writerow(["surname", "", e, n])


def _prompts() -> None:
    import pandas as pd
    cfg = load_config()
    trials = pd.read_parquet(ROOT / "data" / "design" / "trials.parquet")
    doctors = pd.read_parquet(ROOT / "data" / "design" / "doctors.parquet")
    t = trials.sort_values("trial_id").iloc[0]
    example = build_prompt(t, doctors[doctors["trial_id"] == t["trial_id"]],
                           cfg, "choice_first", "card")
    parts = ["# Prompts, personas, and response formats",
             "", "Exact strings from `src/stimuli.py` and `config/experiment.yaml`.",
             "", "## Personas", ""]
    for k, v in cfg["personas"].items():
        parts += [f"- **{k}**: {v.strip()}"]
    parts += [f"- **neutral** (persona-free arm): {NEUTRAL_PERSONA}", "",
              "## Templates", ""]
    for i, tpl in enumerate(TEMPLATES):
        parts += [f"### Template {i}", "", "```", tpl, "```", ""]
    parts += ["## Response-format instructions", ""]
    for k, v in FORMAT_INSTRUCTIONS.items():
        parts += [f"- **{k}**: `{v}`"]
    parts += ["", f"Retry reminder: `{RETRY_REMINDER}`", "",
              f"Manipulation check: `{MANIPULATION_CHECK}`", "",
              f"## Example rendered prompt (main arm, trial {int(t['trial_id'])})",
              "", "```", example, "```", ""]
    (OUT / "docs" / "prompts.md").write_text("\n".join(parts), encoding="utf-8")


PROVENANCE = """# Provenance, prespecification, and deviations

All times UTC, from file metadata and the per-call `ts` field of the raw
records.

- 2026-07-02 07:02 - design generated (seed 20260702); SHA-256 of
  `trials.parquet` and `doctors.parquet` recorded in `data/design/design_meta.json`.
- 2026-07-02 07:14 - first non-pilot model call (llama3.2:3b, main arm).
- 2026-07-02 - `docs/analysis_plan.md` written; final modification 12:10,
  after main-arm collection for llama3.2:3b and qwen2.5:3b had finished and
  while phi3:mini was being collected, before any main-arm analysis.
- 2026-07-02 - panel extended (gemma3:4b, llama3.1:8b, deepseek-r1:7b with a
  5% parse-failure pilot gate), before any main-arm analysis.
- 2026-07-03 - gpt-4o-mini collected (Azure OpenAI); deepseek-r1:7b fails its
  pilot gate (100% parse failures under the 220-token output budget).
- 2026-07-04 - remaining open-weight runs finish; analysis dataset built and
  prespecified analyses run.

`docs/analysis_plan.md` is released verbatim. It says it would be frozen on
OSF before any non-pilot call and labels analyses "preregistered"; it was
**not** deposited in any public registry, so read "preregistered" there as
"prespecified". The plan's power analysis assumed a twelve-model panel; the
realized panel has seven models.

Deviations (see Supplementary Table S1 of the paper):

1. No public registration (above).
2. Seven-model panel instead of twelve (frontier APIs not run under the
   protocol; deepseek-r1:7b excluded at its gate).
3. Holm correction was applied over H1-H13 in code; the plan heading and the
   submitted table note said H1-H12. H13 raw p = .0495, Holm p = .099.
4. Stated reasons were coded with the keyword dictionary only; the judge-model
   coder (`reasons.py --judge`) was not run and the dictionary was not
   hand-validated.
5. Fee-equivalents are reported as fee-range indices (value / $100) because
   the fee response is nonlinear.
6. Reviewer-requested post hoc analyses: `src/analysis/revision_r1.py`
   (outputs in `results/tables/revision/`).
7. The Claude-family session pilot (`data/raw_session_pilot/`) was collected
   inside an agentic coding harness, not under the protocol; it enters no
   estimate. Pilot-based forecasts described in the submitted manuscript
   could not be located and are not included.
"""

README = """# Whose doctor does the AI recommend? - data and code (revision R1)

Replication package for a randomized choice-based conjoint audit of large
language model physician recommendation (International Journal of Medical
Informatics, manuscript IJMEDI-D-26-04763, revision 1).

## Contents
- `config/experiment.yaml` - frozen experiment configuration (seed 20260702)
- `data/design/` - frozen design matrices and their SHA-256 hashes
- `data/raw/` - 42,441 raw model call records (JSONL, one record per call,
  including parse failures, pilots, and the gate-excluded deepseek-r1:7b pilot)
- `data/processed/choices.parquet` - long-format analysis dataset built by
  `src.analysis.prep`
- `data/raw_session_pilot/` - 200 exploratory Claude-family pilot records
  collected inside an agentic coding harness, NOT under the audit protocol;
  excluded from every estimate (see its README_DATA_PROVENANCE.txt)
- `docs/analysis_plan.md` - the prespecified analysis plan, verbatim
- `docs/PROVENANCE.md` - timeline and deviations from the plan
- `docs/prompts.md` - personas, the nine templates, response formats, and an
  example rendered prompt
- `docs/name_lists.csv` - first names per gender x ethnicity cell and surnames
  per ethnicity
- `docs/model_digests.json` - model identifiers and weight digests
- `docs/COAUTHOR_RUNBOOK.md` - instructions for running API models
- `results/` - every table and figure (`results/tables/revision/` holds the
  reviewer-requested post hoc analyses)
- `src/` - the full pipeline: design generation, execution, parsing, analysis

## Reproducing the analysis (no model calls)
    python -m venv .venv
    .venv/bin/pip install -r requirements.txt     (Windows: .venv\\Scripts\\pip)
    python -m src.reproduce

`src.reproduce` rebuilds `data/processed/choices.parquet` from `data/raw/`
and regenerates every table and figure. `python -m src.smoke_test` validates
the pipeline against a mock data-generating process.

## Auditing a new model against the identical stimuli
Add the model to `config/experiment.yaml`, then

    python -m src.run_night --models <model-id>
    python -m src.reproduce

The runner checkpoints every call and is resume-safe. Local models run via
Ollama; API models via the providers configured in `experiment.yaml`
(see `docs/COAUTHOR_RUNBOOK.md`).

## Licenses
Code: MIT (`LICENSE`). Data, design, and results: CC BY 4.0 (`LICENSE-DATA`).
All physician identities are synthetic.
"""

LICENSE_DATA = """The data, design matrices, and results in this package are licensed under
the Creative Commons Attribution 4.0 International License (CC BY 4.0).
https://creativecommons.org/licenses/by/4.0/
All physician identities are synthetic.
"""


def _zip() -> Path:
    z = OUT.with_suffix(".zip")
    if z.exists():
        z.unlink()
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(OUT.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(OUT.parent))
    return z


def main() -> None:
    _copy()
    _digests()
    _names()
    _prompts()
    (OUT / "docs" / "PROVENANCE.md").write_text(PROVENANCE, encoding="utf-8")
    (OUT / "README.md").write_text(README, encoding="utf-8")
    (OUT / "LICENSE-DATA").write_text(LICENSE_DATA, encoding="utf-8")
    z = _zip()
    n = sum(1 for p in OUT.rglob("*") if p.is_file())
    h = hashlib.sha256(z.read_bytes()).hexdigest()
    print(f"[release] {n} files -> {z} ({z.stat().st_size / 1e6:.1f} MB)")
    print(f"[release] sha256 {h}")


if __name__ == "__main__":
    main()
