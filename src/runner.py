"""Execute audit trials against local (Ollama) or API (OpenAI-compatible) models.

Resume-safe: every completed call is appended to a JSONL checkpoint under
data/raw/; rerunning skips already-completed (trial_id, rep) pairs.

Usage examples:
    python -m src.runner --model llama3.2:3b --arm main --pilot
    python -m src.runner --model llama3.2:3b --arm main
    python -m src.runner --model llama3.2:3b --arms main,temp0,retest --wait-gpu
    python -m src.runner --model gpt-4o-mini --arm main          (needs API key env)
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import yaml

try:  # trust the Windows cert store (required when AV intercepts HTTPS)
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

try:  # optional: only needed for provider "anthropic"
    import anthropic as _anthropic
    _ANTHROPIC_ERRORS: tuple = (_anthropic.APIError,)
except ImportError:
    _anthropic = None
    _ANTHROPIC_ERRORS = ()

# Errors treated as transient by the per-call retry loop.
RETRYABLE_ERRORS = (requests.RequestException, KeyError) + _ANTHROPIC_ERRORS

from src import stimuli

ROOT = Path(__file__).resolve().parents[1]
OLLAMA_URL = "http://localhost:11434"


# --------------------------------------------------------------------------
# Config / design loading
# --------------------------------------------------------------------------

def load_config() -> dict:
    """Load experiment config; AUDIT_CONFIG env var overrides the path."""
    import os
    path = os.environ.get("AUDIT_CONFIG") or (ROOT / "config" / "experiment.yaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_design(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = ROOT / cfg["paths"]["design"]
    return pd.read_parquet(d / "trials.parquet"), pd.read_parquet(d / "doctors.parquet")


def model_entry(cfg: dict, model_id: str) -> dict:
    for group in ("local", "frontier"):
        for m in cfg["models"][group]:
            if m["id"] == model_id:
                return m
    if model_id == cfg["reason_judge"]["id"]:
        return cfg["reason_judge"]
    raise SystemExit(f"Model {model_id!r} not found in config")


def call_seed(model: str, arm: str, trial_id: int, rep: int) -> int:
    h = hashlib.md5(f"{model}|{arm}|{trial_id}|{rep}".encode()).hexdigest()
    return int(h[:8], 16)


# --------------------------------------------------------------------------
# Clients
# --------------------------------------------------------------------------

class OllamaClient:
    def __init__(self, model: str, gen: dict):
        self.model, self.gen = model, gen

    def chat(self, prompt: str, temperature: float, seed: int,
             json_mode: bool, want_logprobs: bool = False) -> dict:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "keep_alive": "15m",
            "options": {
                "temperature": temperature,
                "seed": seed,
                "top_p": self.gen["top_p"],
                "num_predict": self.gen["max_tokens"],
            },
        }
        if json_mode:
            payload["format"] = "json"
        if want_logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = 5
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload,
                          timeout=self.gen["timeout_s"])
        r.raise_for_status()
        data = r.json()
        return {
            "text": data["message"]["content"],
            "logprobs": data["message"].get("logprobs"),
            "eval_count": data.get("eval_count"),
            "prompt_eval_count": data.get("prompt_eval_count"),
        }


class OpenAICompatClient:
    def __init__(self, model: str, gen: dict, base_url: str, api_key: str,
                 omit_sampling: bool = False):
        self.model, self.gen = model, gen
        self.base_url, self.api_key = base_url.rstrip("/"), api_key
        # Some models (e.g. OpenAI reasoning-series) reject temperature/top_p/
        # seed; set omit_sampling: true on the config entry for those. The
        # temp0 arm is then meaningless for that model and is skipped in
        # analysis (single-temperature model; note in design_meta).
        self.omit_sampling = omit_sampling

    def chat(self, prompt: str, temperature: float, seed: int,
             json_mode: bool, want_logprobs: bool = False) -> dict:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.gen["max_tokens"],
        }
        if not self.omit_sampling:
            payload.update(temperature=temperature, seed=seed,
                           top_p=self.gen["top_p"])
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if want_logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = 5
        r = requests.post(f"{self.base_url}/chat/completions", json=payload,
                          headers={"Authorization": f"Bearer {self.api_key}"},
                          timeout=self.gen["timeout_s"])
        r.raise_for_status()
        data = r.json()
        ch = data["choices"][0]
        return {
            "text": ch["message"]["content"],
            "logprobs": ch.get("logprobs"),
            "eval_count": data.get("usage", {}).get("completion_tokens"),
            "prompt_eval_count": data.get("usage", {}).get("prompt_tokens"),
        }


class AnthropicClient:
    """Native Anthropic Messages API client (official SDK).

    Two audited-panel caveats, both recorded in the runbook and design meta:
    the Anthropic API has no seed parameter (the runner's call seed is still
    logged in checkpoints for bookkeeping, but calls are not seed-reproducible),
    and Claude 4+ accepts at most one of temperature/top_p, so this client
    sends temperature only. JSON output is elicited by the prompt's format
    instruction (as for every provider) with the runner's parse-retry loop.
    No logprobs support: the letter arm records sampled choices only.
    """

    def __init__(self, model: str, gen: dict, api_key: str):
        if _anthropic is None:
            raise SystemExit("pip install anthropic (required for provider "
                             "'anthropic')")
        self.model, self.gen = model, gen
        self.client = _anthropic.Anthropic(
            api_key=api_key, timeout=float(gen["timeout_s"]), max_retries=0)

    def chat(self, prompt: str, temperature: float, seed: int,
             json_mode: bool, want_logprobs: bool = False) -> dict:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.gen["max_tokens"],
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        return {
            "text": text,
            "logprobs": None,
            "eval_count": resp.usage.output_tokens if resp.usage else None,
            "prompt_eval_count": resp.usage.input_tokens if resp.usage else None,
        }


class AzureOpenAIClient:
    def __init__(self, model: str, gen: dict, endpoint: str, api_key: str,
                 api_version: str, deployment: str):
        self.model, self.gen = model, gen
        self.endpoint, self.api_key = endpoint, api_key
        self.api_version, self.deployment = api_version, deployment

    def chat(self, prompt: str, temperature: float, seed: int,
             json_mode: bool, want_logprobs: bool = False) -> dict:
        from openai import AzureOpenAI
        client = AzureOpenAI(api_key=self.api_key, azure_endpoint=self.endpoint,
                             api_version=self.api_version)
        kwargs = dict(
            model=self.deployment,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            seed=seed,
            max_tokens=self.gen["max_tokens"],
            top_p=self.gen["top_p"],
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if want_logprobs:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = 5
        r = client.chat.completions.create(**kwargs)
        ch = r.choices[0]
        return {
            "text": ch.message.content,
            "logprobs": ch.logprobs.model_dump() if ch.logprobs else None,
            "eval_count": r.usage.completion_tokens if r.usage else None,
            "prompt_eval_count": r.usage.prompt_tokens if r.usage else None,
        }


def make_client(cfg: dict, model_id: str):
    import os
    entry = model_entry(cfg, model_id)
    if entry["provider"] == "mock":
        from src.mock_client import MockClient
        return MockClient(model_id, cfg["generation"])
    if entry["provider"] == "ollama":
        return OllamaClient(model_id, cfg["generation"])
    if entry["provider"] == "azure":
        key = os.environ.get(entry["api_key_env"], "")
        endpoint = os.environ.get(entry.get("endpoint_env", ""), "") or entry.get("endpoint", "")
        deployment = os.environ.get(entry.get("deployment_env", ""), "") or entry.get("deployment", model_id)
        api_version = entry.get("api_version", "2025-01-01-preview")
        if not key:
            raise SystemExit(f"Environment variable {entry['api_key_env']} is not set")
        if not endpoint:
            raise SystemExit(f"Azure endpoint not set (env: {entry.get('endpoint_env')})")
        return AzureOpenAIClient(model_id, cfg["generation"], endpoint, key, api_version, deployment)
    key = os.environ.get(entry["api_key_env"], "")
    if not key:
        raise SystemExit(f"Environment variable {entry['api_key_env']} is not set")
    if entry["provider"] == "anthropic":
        return AnthropicClient(model_id, cfg["generation"], key)
    return OpenAICompatClient(model_id, cfg["generation"], entry["base_url"], key,
                              omit_sampling=bool(entry.get("omit_sampling")))


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

def extract_json(text: str) -> dict | None:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                return None
    return None


def match_name(value: str, names: list[str]) -> str | None:
    if not isinstance(value, str):
        return None

    def core(s: str) -> str:  # strip a "Dr." / "Dr" title prefix
        s = s.strip().casefold()
        for pre in ("dr.", "dr "):
            if s.startswith(pre):
                s = s[len(pre):].strip()
        return s

    v = core(value)
    for n in names:
        if v == n.casefold() or v == core(n):
            return n
    # bare surname ("Patel" or "Dr. Patel" -> "Dr. Priya Patel") if unambiguous
    hits = [n for n in names if v == core(n).split()[-1]]
    if len(hits) == 1:
        return hits[0]
    close = difflib.get_close_matches(value.strip(), names, n=1, cutoff=0.8)
    return close[0] if close else None


def parse_response(text: str, response_kind: str, names: list[str],
                   letters: dict[str, str]) -> dict:
    """Returns dict with parsed fields + ok flag."""
    if response_kind == "letter":
        m = re.search(r"\b([A-E])\b", text.strip())
        if m:
            return {"ok": True, "choice": letters[m.group(1)], "reason": None,
                    "ranking": None}
        return {"ok": False, "choice": None, "reason": None, "ranking": None}

    obj = extract_json(text)
    if obj is None:
        return {"ok": False, "choice": None, "reason": None, "ranking": None}

    if response_kind == "rank3":
        ranking_raw = obj.get("ranking", [])
        if not isinstance(ranking_raw, list):
            return {"ok": False, "choice": None, "reason": obj.get("reason"),
                    "ranking": None}
        ranked = [match_name(x, names) for x in ranking_raw[:3]]
        if None in ranked or len(set(ranked)) != 3:
            return {"ok": False, "choice": None, "reason": obj.get("reason"),
                    "ranking": None}
        return {"ok": True, "choice": ranked[0], "reason": obj.get("reason"),
                "ranking": ranked}

    choice = match_name(obj.get("choice"), names)
    return {"ok": choice is not None, "choice": choice,
            "reason": obj.get("reason"), "ranking": None}


# --------------------------------------------------------------------------
# GPU guard
# --------------------------------------------------------------------------

def gpu_is_free(cfg: dict) -> bool:
    g = cfg["gpu_guard"]
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20).stdout.strip()
        util, mem = (int(x) for x in out.split(","))
        return util <= g["max_util_pct"] and mem <= g["max_mem_mb"]
    except Exception:
        return True  # no nvidia-smi -> nothing to guard


def wait_for_gpu(cfg: dict) -> None:
    g = cfg["gpu_guard"]
    streak = 0
    while streak < g["checks"]:
        if gpu_is_free(cfg):
            streak += 1
        else:
            streak = 0
        if streak < g["checks"]:
            print(f"[gpu-guard] waiting (free-streak {streak}/{g['checks']})")
            time.sleep(g["interval_s"])
    print("[gpu-guard] GPU is free, starting")


# --------------------------------------------------------------------------
# Trial execution
# --------------------------------------------------------------------------

def arm_trials(cfg: dict, trials: pd.DataFrame, arm: str, pilot: bool) -> pd.DataFrame:
    if arm == "main":
        sel = trials
    elif arm == "manip":
        return trials.head(0)  # handled separately
    else:
        sel = trials[trials[f"arm_{arm}"]]
    sel = sel.sort_values("run_order")
    if pilot:
        sel = sel.head(cfg["pilot"]["n_trials"])
    return sel


def checkpoint_path(cfg: dict, model: str, arm: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", model)
    p = ROOT / cfg["paths"]["raw"] / f"{safe}__{arm}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def completed_keys(path: Path) -> set[tuple[int, int]]:
    done = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done.add((rec["trial_id"], rec["rep"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def run_manipulation_check(cfg: dict, client, model: str, ckpt: Path,
                           done: set) -> None:
    for rep in range(20):
        if (-1, rep) in done:
            continue
        seed = call_seed(model, "manip", -1, rep)
        res = client.chat(stimuli.MANIPULATION_CHECK, 0.7, seed, json_mode=False)
        rec = dict(trial_id=-1, rep=rep, arm="manip", model=model, seed=seed,
                   raw=res["text"], ok=True, choice=None, reason=None,
                   ranking=None, attempts=1,
                   ts=datetime.now(timezone.utc).isoformat())
        with open(ckpt, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    print(f"[{model}] manipulation check complete")


def run_arm(cfg: dict, model: str, arm: str, pilot: bool,
            workers: int = 1) -> None:
    trials, doctors = load_design(cfg)
    client = make_client(cfg, model)
    ckpt = checkpoint_path(cfg, model, arm + ("_pilot" if pilot else ""))
    done = completed_keys(ckpt)

    if arm == "manip":
        run_manipulation_check(cfg, client, model, ckpt, done)
        return

    spec = cfg["arms"][arm]
    response = spec["response"]
    fmt = spec["format"]
    temperature = spec["temperature"]
    reps = spec.get("reps", 1)
    json_mode = response != "letter"
    want_logprobs = bool(spec.get("requires_logprobs"))
    persona_override = spec.get("persona_override")

    sel = arm_trials(cfg, trials, arm, pilot)
    todo = [(t, r) for _, t in sel.iterrows() for r in range(reps)
            if (t["trial_id"], r) not in done]
    if cfg.get("_max_trials"):
        todo = todo[:cfg["_max_trials"]]
    print(f"[{model}|{arm}] {len(todo)} calls to run "
          f"({len(done)} already complete, workers={workers})")

    lock = threading.Lock()
    state = {"n": 0, "fail": 0}
    t0 = time.time()

    def do_call(item: tuple) -> None:
        trial, rep = item
        tid = int(trial["trial_id"])
        doc = doctors[doctors["trial_id"] == tid]
        names = doc.sort_values("slot")["name"].tolist()
        letters = dict(zip(doc["letter"], doc["name"]))
        prompt = stimuli.build_prompt(trial, doc, cfg, response, fmt,
                                      persona_override=persona_override)
        seed = call_seed(model, arm, tid, rep)

        parsed, raw, attempts, res = None, "", 0, {}
        for attempt in range(1 + cfg["generation"]["retries"]):
            attempts = attempt + 1
            p = prompt if attempt == 0 else (
                prompt + "\n\n" + stimuli.RETRY_REMINDER.format(
                    names=", ".join(names)))
            try:
                res = client.chat(p, temperature, seed + attempt,
                                  json_mode, want_logprobs)
            except RETRYABLE_ERRORS as e:
                time.sleep(min(30, 2 ** attempt * 5))
                raw = f"<request error: {e}>"
                continue
            raw = res["text"]
            parsed = parse_response(raw, response, names, letters)
            if parsed["ok"]:
                break
        if parsed is None:
            parsed = {"ok": False, "choice": None, "reason": None, "ranking": None}

        chosen_slot = None
        if parsed["choice"] is not None:
            chosen_slot = int(doc.loc[doc["name"] == parsed["choice"], "slot"].iloc[0])

        rec = dict(trial_id=tid, rep=rep, arm=arm, model=model, seed=seed,
                   raw=raw, ok=parsed["ok"], choice=parsed["choice"],
                   chosen_slot=chosen_slot, reason=parsed["reason"],
                   ranking=parsed["ranking"], attempts=attempts,
                   logprobs=res.get("logprobs") if parsed["ok"] and want_logprobs else None,
                   ts=datetime.now(timezone.utc).isoformat())
        with lock:
            with open(ckpt, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            state["n"] += 1
            if not parsed["ok"]:
                state["fail"] += 1
            n = state["n"]
            if n % 25 == 0 or n == len(todo):
                rate = (time.time() - t0) / n
                eta_min = rate * (len(todo) - n) / 60
                print(f"[{model}|{arm}] {n}/{len(todo)} "
                      f"fail={state['fail']} avg={rate:.1f}s eta={eta_min:.0f}min",
                      flush=True)

    if workers <= 1:
        for item in todo:
            do_call(item)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(do_call, todo))

    total = len(todo) if todo else 1
    print(f"[{model}|{arm}] DONE. parse-failure rate: "
          f"{state['fail']}/{len(todo)} ({100 * state['fail'] / total:.1f}%)")


def probe_logprobs(cfg: dict, model: str) -> bool:
    try:
        client = make_client(cfg, model)
        res = client.chat("Reply with the single letter A.", 0.0, 1,
                          json_mode=False, want_logprobs=True)
        ok = res.get("logprobs") is not None
        print(f"[{model}] logprobs supported: {ok}")
        return ok
    except Exception as e:
        print(f"[{model}] logprobs probe failed: {e}")
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", default=None)
    ap.add_argument("--arms", default=None, help="comma-separated list")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--wait-gpu", action="store_true")
    ap.add_argument("--probe-logprobs", action="store_true")
    ap.add_argument("--max-trials", type=int, default=0,
                    help="stop after N trials (0 = no limit)")
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent requests (server must allow parallel; "
                         "use with OLLAMA_NUM_PARALLEL>=workers)")
    args = ap.parse_args()

    cfg = load_config()
    if args.max_trials:
        cfg["_max_trials"] = args.max_trials
    if args.probe_logprobs:
        probe_logprobs(cfg, args.model)
        return
    if args.wait_gpu and model_entry(cfg, args.model)["provider"] == "ollama":
        wait_for_gpu(cfg)

    arms = (args.arms.split(",") if args.arms else [args.arm])
    for arm in arms:
        if not arm:
            raise SystemExit("Provide --arm or --arms")
        run_arm(cfg, args.model, arm.strip(), args.pilot,
                workers=max(1, args.workers))


if __name__ == "__main__":
    main()
