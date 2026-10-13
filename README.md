# Whose doctor does the AI recommend? - data and code

Replication package for a randomized choice-based conjoint audit of large
language model physician recommendation (International Journal of Medical
Informatics, manuscript IJMEDI-D-26-04763)

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
    .venv/bin/pip install -r requirements.txt     (Windows: .venv\Scripts\pip)
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
