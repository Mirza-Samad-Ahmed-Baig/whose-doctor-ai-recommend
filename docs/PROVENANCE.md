# Provenance, prespecification, and deviations

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
