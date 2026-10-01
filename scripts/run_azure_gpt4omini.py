"""Detached runner: gpt-4o-mini (Azure) pilot then full protocol.

Reads Azure credentials from the environment (inherited from launcher).
Run detached: .venv\Scripts\python.exe scripts\run_azure_gpt4omini.py
"""
import os, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
LOG = ROOT / "logs"

for var in ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_DEPLOYMENT"):
    if not os.environ.get(var):
        raise SystemExit(f"{var} not set")

def run(args):
    with open(LOG / "azure_stdout.log", "ab") as out:
        subprocess.run(args, cwd=str(ROOT), env=os.environ.copy(),
                       stdout=out, stderr=subprocess.STDOUT)

run([str(PY), "-m", "src.run_night", "--no-wait-gpu", "--pilot",
     "--models", "gpt-4o-mini"])
(LOG / "AZURE_PILOT_DONE.marker").write_text("done")
run([str(PY), "-m", "src.run_night", "--no-wait-gpu", "--workers", "3",
     "--models", "gpt-4o-mini"])
(LOG / "AZURE_DONE.marker").write_text("done")
