# Co-author runbook — frontier-model arm of the physician audit

You are running the **frontier API panel** (Claude, Gemini, GPT) against the
same frozen design the local open-weight panel used. Your API keys never leave
your machine; you send back only the raw JSONL output files.

Total workload per model: **5,744 calls** (~700 input + ~220 output tokens
each). Rough cost: Claude Sonnet 4.6 ≈ $30–35, Gemini 2.5 Flash ≈ $3–6,
GPT flagship ≈ $15–25. Runtime: 2–6 h per model at 1 worker (API-bound, no
GPU needed; a laptop is fine). Everything is checkpointed and resume-safe —
you can stop and restart at any point.

## 1. One-time setup (~10 minutes)

1. Install **Python 3.12** (python.org, check "Add to PATH").
2. Unzip the project archive anywhere (e.g. `C:\physician-audit\` or
   `~/physician-audit/`). Keep the folder structure intact.
3. In a terminal at the project root:

   ```powershell
   py -3.12 -m venv .venv
   .venv\Scripts\python -m pip install -r requirements.txt
   ```

   (macOS/Linux: `python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt`)

4. Set your API keys as environment variables (only the ones you have):

   ```powershell
   # Windows (persistent; open a NEW terminal afterwards)
   setx ANTHROPIC_API_KEY "sk-ant-..."
   setx GEMINI_API_KEY    "AIza..."
   setx OPENAI_API_KEY    "sk-..."
   ```

   ```bash
   # macOS/Linux (add to ~/.zshrc or ~/.bashrc)
   export ANTHROPIC_API_KEY="sk-ant-..."
   export GEMINI_API_KEY="AIza..."
   export OPENAI_API_KEY="sk-..."
   ```

5. Enable the models you'll run: in `config/experiment.yaml`, under
   `models: frontier:`, flip `enabled: false` → `enabled: true` for
   `claude-sonnet-4-6`, `gemini-2.5-flash`, and/or the OpenAI entry.
   For OpenAI, first check which model your key can access
   (`https://api.openai.com/v1/models`) and put that exact id in the config.

6. **Verify the design is byte-identical to ours** (critical for the paper's
   common-stimuli claim):

   ```powershell
   .venv\Scripts\python -c "import hashlib, json, pathlib; meta = json.loads(pathlib.Path('data/design/design_meta.json').read_text()); ok = all(hashlib.sha256(pathlib.Path(f'data/design/{n}.parquet').read_bytes()).hexdigest() == h for n, h in meta['sha256'].items()); print('DESIGN HASH OK' if ok else 'HASH MISMATCH - STOP, tell Mirza')"
   ```

## 2. Pilot each model first (~1 minute each)

```powershell
.venv\Scripts\python -m src.runner --model claude-sonnet-4-6 --arm main --pilot --max-trials 10
```

Expect `parse-failure rate: 0/10` or close. If it errors immediately, the key
or model id is wrong — nothing has been spent. Repeat for each model.

## 3. Full runs

One command per model (runs all nine arms + manipulation check, sequentially):

```powershell
.venv\Scripts\python -m src.runner --model claude-sonnet-4-6 --arms main,temp0,grounded,rank_top3,field_order,retest,letter,nopersona,card_reversed,manip
.venv\Scripts\python -m src.runner --model claude-haiku-4-5  --arms main,temp0,grounded,rank_top3,field_order,retest,letter,nopersona,card_reversed,manip
.venv\Scripts\python -m src.runner --model gemini-2.5-flash  --arms main,temp0,grounded,rank_top3,field_order,retest,letter,nopersona,card_reversed,manip
.venv\Scripts\python -m src.runner --model <openai-model-id> --arms main,temp0,grounded,rank_top3,field_order,retest,letter,nopersona,card_reversed,manip
```

`claude-haiku-4-5` is a cheap (~$10) but high-value addition: it is the only
API-runnable target of the session-pilot predictions in
`predicted_from_pilot/`, so running it enables the pilot-forecast vs
measured-run methods comparison.

Notes:
- Progress prints every 25 calls with an ETA. Ctrl-C any time; rerunning the
  same command resumes where it stopped.
- If a provider rate-limits you, the runner backs off and retries; sustained
  429s just slow it down.
- **OpenAI only:** if the run fails instantly with an error about
  `temperature`/`top_p` being unsupported, add `omit_sampling: true` to that
  model's config line, rerun, and tell Mirza — that model then skips the
  temp0 arm in analysis.
- Models can run in parallel terminals (they're different providers).

## 4. Send back the data

When a model finishes, its output lives in `data/raw/` as
`<model>__<arm>.jsonl` files (about 10–20 MB per model, plus `*_pilot.jsonl`
files — send those too). Zip the whole `data/raw/` folder and send it over.
Nothing in it contains your API keys.

## 5. Known provider quirks (already handled — for your awareness)

- **Anthropic:** no `seed` parameter (calls aren't seed-reproducible; the
  retest arm still measures consistency) and no logprobs (the letter arm
  records sampled choices only). The client sends `temperature` but not
  `top_p` (Claude 4+ accepts only one). We deliberately audit
  `claude-sonnet-4-6` rather than Sonnet 5 / Opus 4.7+ because those reject
  non-default temperature, which the design's temperature arms require.
- **Gemini:** runs through Google's OpenAI-compatible endpoint; nothing
  special to do.
- Record the exact model ids and run dates — they go in the paper's methods
  section (`design_meta` note).
