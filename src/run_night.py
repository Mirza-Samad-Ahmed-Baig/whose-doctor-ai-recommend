"""One-command overnight orchestrator for the local-model audit runs.

Sequences every local (Ollama) model in the config through the full arm list
via subprocess calls to ``src.runner``. Intended to be launched once, after the
user frees the GPU, and left to run overnight. Resume-safety is inherited from
the runner's JSONL checkpoints: this orchestrator holds no state of its own, so
it is safe to re-invoke after any crash or interrupt -- already-completed
(trial, rep) calls are skipped and the summary is re-derived from the
checkpoints on disk.

Per-model console output is tee'd to ``logs/{model}__{timestamp}.log``. When all
runs finish, a per-(model, arm) summary is printed and written to
``logs/night_summary.csv`` (columns: model, arm, calls_done, parse_fail_rate,
minutes).

Examples:
    python -m src.run_night                       # all local models, full arms
    python -m src.run_night --pilot               # 50-trial pilot per model
    python -m src.run_night --models llama3.2:3b,qwen2.5:3b
    python -m src.run_night --no-wait-gpu         # skip the GPU idle guard

The GPU guard (--wait-gpu) is ON by default and passed through to the runner,
which defers each model's start until the GPU has been idle for the configured
streak. This script never touches the GPU itself.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.design import load_config

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

ARMS = ["main", "temp0", "grounded", "rank_top3", "field_order", "retest",
        "letter", "nopersona", "card_reversed", "manip"]


def local_models(cfg: dict) -> list[str]:
    """All models under the config's ``local`` panel (any provider).

    For the real config these are the Ollama models; the runner decides how to
    talk to each, and its --wait-gpu guard only fires for the ollama provider,
    so the orchestrator stays provider-agnostic.
    """
    ids = [m["id"] for m in cfg["models"]["local"]]
    ids += [m["id"] for m in cfg["models"].get("frontier", [])
            if m.get("enabled")]
    return ids


def safe_name(model: str) -> str:
    """Match src.runner.checkpoint_path's sanitization of model ids."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", model)


def checkpoint_file(cfg: dict, model: str, arm: str, pilot: bool) -> Path:
    suffix = "_pilot" if pilot else ""
    return (ROOT / cfg["paths"]["raw"]
            / f"{safe_name(model)}__{arm}{suffix}.jsonl")


def arm_stats(path: Path) -> tuple[int, float, float]:
    """Return (calls_done, parse_fail_rate, span_minutes) from a checkpoint.

    span_minutes is the wall span between the first and last recorded call
    timestamps in this arm's checkpoint (the actual generation span; robust to
    resumes only in that it reflects recorded data, not orchestration time).
    """
    if not path.exists():
        return 0, 0.0, 0.0
    n = 0
    n_fail = 0
    ts_min: datetime | None = None
    ts_max: datetime | None = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            if not rec.get("ok", False):
                n_fail += 1
            ts = rec.get("ts")
            if ts:
                try:
                    t = datetime.fromisoformat(ts)
                except ValueError:
                    t = None
                if t is not None:
                    ts_min = t if ts_min is None or t < ts_min else ts_min
                    ts_max = t if ts_max is None or t > ts_max else ts_max
    rate = round(n_fail / n, 4) if n else 0.0
    minutes = round((ts_max - ts_min).total_seconds() / 60, 1) \
        if ts_min and ts_max else 0.0
    return n, rate, minutes


def workers_for(model: str, requested: int) -> int:
    """7B-class models exceed the 4GB VRAM budget already; keep them serial."""
    return 1 if re.search(r"7b", model, re.IGNORECASE) else max(1, requested)


def run_model(cfg: dict, model: str, pilot: bool, wait_gpu: bool,
              workers: int = 1) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"{safe_name(model)}__{ts}.log"

    cmd = [PY, "-m", "src.runner", "--model", model, "--arms", ",".join(ARMS),
           "--workers", str(workers_for(model, workers))]
    if pilot:
        cmd.append("--pilot")
    if wait_gpu:
        cmd.append("--wait-gpu")

    print(f"[night] starting {model}  (pilot={pilot}, wait_gpu={wait_gpu})")
    print(f"[night]   log -> {log_path}")
    t0 = time.time()
    # Stream child output to the log file and mirror it to our stdout so the
    # operator sees progress live; line-buffered, resume-safe (append).
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write(f"# {model} | started {ts} | cmd: {' '.join(cmd)}\n")
        lf.flush()
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            lf.write(line)
            lf.flush()
            sys.stdout.write(f"  [{model}] {line}")
        proc.wait()
    dt = (time.time() - t0) / 60
    status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
    print(f"[night] {model} {status} in {dt:.1f} min")


def build_summary(cfg: dict, models: list[str], pilot: bool) -> pd.DataFrame:
    rows = []
    for model in models:
        for arm in ARMS:
            calls, rate, minutes = arm_stats(
                checkpoint_file(cfg, model, arm, pilot))
            rows.append(dict(model=model, arm=arm, calls_done=calls,
                             parse_fail_rate=rate, minutes=minutes))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Overnight orchestrator: run all local models x all arms "
                    "via src.runner, with checkpoint-based resume safety.")
    ap.add_argument("--pilot", action="store_true",
                    help="50-trial pilot mode (passed through to the runner)")
    ap.add_argument("--models", default=None,
                    help="comma-separated subset of local model ids to run "
                         "(default: all Ollama models in the config)")
    ap.add_argument("--wait-gpu", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="defer each model until the GPU is idle "
                         "(default: on; use --no-wait-gpu to disable)")
    ap.add_argument("--summary-only", action="store_true",
                    help="skip running; just rebuild logs/night_summary.csv "
                         "from existing checkpoints")
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent requests per model (3B-class only; "
                         "7B models always run serial). Requires the Ollama "
                         "server to run with OLLAMA_NUM_PARALLEL>=workers.")
    args = ap.parse_args()

    cfg = load_config()
    available = local_models(cfg)
    if args.models:
        wanted = [m.strip() for m in args.models.split(",") if m.strip()]
        unknown = [m for m in wanted if m not in available]
        if unknown:
            raise SystemExit(f"Unknown local model(s): {unknown}. "
                             f"Available: {available}")
        models = wanted
    else:
        models = available
    if not models:
        raise SystemExit("No local Ollama models found in config.")

    if not args.summary_only:
        print(f"[night] models: {models}")
        print(f"[night] arms:   {','.join(ARMS)}")
        overall = time.time()
        for model in models:
            run_model(cfg, model, args.pilot, args.wait_gpu,
                      workers=args.workers)
        print(f"[night] all runs done in {(time.time() - overall) / 60:.1f} min")

    summary = build_summary(cfg, models, args.pilot)
    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    out = logs / "night_summary.csv"
    summary.to_csv(out, index=False)
    print(f"\n[night] summary -> {out}")
    print(summary.to_string(index=False))

    flagged = summary[summary["parse_fail_rate"] > 0.05]
    if not flagged.empty:
        print("\n[night] WARNING: arms with >5% parse failure "
              "(see preregistration exclusion rule):")
        print(flagged.to_string(index=False))


if __name__ == "__main__":
    main()
