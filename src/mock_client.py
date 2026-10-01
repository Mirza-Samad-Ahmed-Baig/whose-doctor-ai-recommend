"""Mock LLM client for the end-to-end pipeline smoke test.

Implements a conditional-logit data-generating process with KNOWN effects:
it parses the rendered physician cards/snippets out of the actual prompt
(which simultaneously validates the stimulus rendering), computes a utility
per doctor, adds Gumbel noise seeded per call, and answers in whatever format
the prompt requests. The analysis pipeline must then recover these effects:

    TRUE_BETAS (logit scale, Gumbel scale = 1)
      rating       +0.45 per level step (3.9 -> 4.3 -> 4.7)
      volume       +0.25 per level step (12 -> 85 -> 400)
      recency      +0.20 (recent)
      responds     +0.15 (practice responds to patient feedback)
      network      +0.15 (university hospital system)
      price        -0.30 per level step (90 -> 140 -> 190)
      telehealth   +0.20
      experience   +0.15 per level step (8 -> 18 -> 28 years)
      female       +0.10 (name-signaled)
      black        -0.15 (name-signaled; hispanic/east_asian/south_asian = 0,
                    so the pipeline must recover true nulls without false
                    positives)
      position 1   +0.30 (first-listed bonus)

The mock's stated reasons NEVER mention gender or ethnicity, mirroring the
expectation that stated rationales under-report demographic drivers; this
exercises the stated-vs-revealed divergence path.

~1.5%% of calls return unparseable text on the first attempt to exercise
the retry path. Never used outside config/experiment_smoke.yaml.
"""

from __future__ import annotations

import hashlib
import json
import math
import re

import numpy as np

from src.design import FIRST_NAMES

TRUE_BETAS = dict(rating=0.45, volume=0.25, recency=0.20, responds=0.15,
                  network=0.15, price=-0.30, telehealth=0.20, experience=0.15,
                  female=0.10, black=-0.15, pos1=0.30)

RATING_CODE = {"3.9": 0, "4.3": 1, "4.7": 2}
VOLUME_CODE = {"12": 0, "85": 1, "400": 2}
PRICE_CODE = {"90": 0, "140": 1, "190": 2}
EXP_CODE = {"8": 0, "18": 1, "28": 2}

# first name -> (gender, ethnicity), inverted from the design's name pools
NAME_CELL = {first: cell for cell, firsts in FIRST_NAMES.items()
             for first in firsts}

REASON_PHRASES = dict(
    rating="the patient rating is excellent",
    volume="there are many patient reviews",
    recency="the reviews are very recent",
    responds="the practice responds to patient feedback",
    network="the doctor is affiliated with a university hospital system",
    price="the visit fee is affordable",
    telehealth="telehealth visits are available",
    experience="the doctor has many years of experience",
)


def _parse_blocks(prompt: str) -> list[dict]:
    """Extract physician attributes from rendered cards or snippets."""
    doctors = []
    for block in prompt.split("\n\n"):
        if "Dr. " not in block or "/5" not in block:
            continue
        name_m = re.search(r"(Dr\. [A-Z][\w'-]+ [A-Z][\w'-]+)", block)
        rating_m = re.search(r"([34]\.\d)/5", block)
        volume_m = re.search(r"(\d+)\s+(?:patient\s+)?reviews", block)
        price_m = re.search(r"\$(\d+)", block)
        exp_m = re.search(r"[Ii]n practice for (\d+) years", block)
        if not (name_m and rating_m and volume_m and price_m and exp_m):
            continue
        first = name_m.group(1).split()[1]
        gender, ethnicity = NAME_CELL[first]
        letter_m = re.search(r"Option ([A-E]):", block)
        doctors.append(dict(
            name=name_m.group(1),
            rating=RATING_CODE[rating_m.group(1)],
            volume=VOLUME_CODE[volume_m.group(1)],
            price=PRICE_CODE[price_m.group(1)],
            experience=EXP_CODE[exp_m.group(1)],
            recent=int("3 days ago" in block),
            responds=int("patient feedback" in block),
            network=int("university hospital system" in block),
            telehealth=int("Telehealth" in block),
            female=int(gender == "female"),
            black=int(ethnicity == "black"),
            letter=letter_m.group(1) if letter_m else None,
        ))
    return doctors


def _utility(d: dict, position: int) -> float:
    b = TRUE_BETAS
    return (b["rating"] * d["rating"] + b["volume"] * d["volume"]
            + b["recency"] * d["recent"] + b["responds"] * d["responds"]
            + b["network"] * d["network"] + b["price"] * d["price"]
            + b["telehealth"] * d["telehealth"]
            + b["experience"] * d["experience"]
            + b["female"] * d["female"] + b["black"] * d["black"]
            + (b["pos1"] if position == 1 else 0.0))


def _top_driver(d: dict) -> str:
    """Largest non-demographic utility contributor (reasons never cite
    gender/ethnicity, by construction)."""
    b = TRUE_BETAS
    contrib = dict(rating=b["rating"] * d["rating"], volume=b["volume"] * d["volume"],
                   recency=b["recency"] * d["recent"], responds=b["responds"] * d["responds"],
                   network=b["network"] * d["network"],
                   price=-b["price"] * (2 - d["price"]),
                   telehealth=b["telehealth"] * d["telehealth"],
                   experience=b["experience"] * d["experience"])
    return max(contrib, key=contrib.get)


class MockClient:
    def __init__(self, model: str, gen: dict):
        self.model, self.gen = model, gen
        # mock-beta weights rating 35% higher: gives cross-model heterogeneity
        self.rating_scale = 1.35 if model.endswith("beta") else 1.0

    def chat(self, prompt: str, temperature: float, seed: int,
             json_mode: bool, want_logprobs: bool = False) -> dict:
        rng = np.random.default_rng(seed)

        if "In one sentence" in prompt:  # manipulation check
            return {"text": "Board certification means a physician has "
                            "completed accredited residency training and "
                            "passed a rigorous specialty examination.",
                    "logprobs": None, "eval_count": 22, "prompt_eval_count": 40}

        doctors = _parse_blocks(prompt)
        if not doctors:
            return {"text": "<mock parse error: no doctors found>",
                    "logprobs": None, "eval_count": 0, "prompt_eval_count": 0}

        # ~1% of choice sets: a PERSISTENT abstention. Keyed on the doctor
        # names (identical across a trial's retry attempts), so it is saved as
        # ok==False with refusal language -- exercising src/analysis/refusals.
        # It never cites a demographic attribute, so the demo-aware rate stays
        # ~0 (the pipeline must not raise a false positive there).
        abstain_key = "".join(sorted(d["name"] for d in doctors))
        if int(hashlib.md5(abstain_key.encode()).hexdigest()[:6], 16) % 100 == 0:
            return {"text": "I cannot recommend one over the others; all five "
                            "physicians appear equally qualified.",
                    "logprobs": None, "eval_count": 16, "prompt_eval_count": 60}

        utils = []
        for i, d in enumerate(doctors):
            u = _utility(d, position=i + 1)
            u += TRUE_BETAS["rating"] * d["rating"] * (self.rating_scale - 1)
            if temperature > 0:
                u += rng.gumbel()
            utils.append(u)
        order = np.argsort(utils)[::-1]
        best = doctors[order[0]]

        # ~1.5% injected first-attempt failures (seed-deterministic; the retry
        # call arrives with seed+1 so the failure does not repeat)
        if seed % 67 == 0:
            return {"text": "Honestly any of these doctors would be fine.",
                    "logprobs": None, "eval_count": 12, "prompt_eval_count": 50}

        letter_mode = "single" in prompt and "capital letter" in prompt
        if letter_mode:
            # softmax over noiseless utilities for fake logprobs
            base = [_utility(d, i + 1) for i, d in enumerate(doctors)]
            mx = max(base)
            probs = [math.exp(u - mx) for u in base]
            z = sum(probs)
            probs = [p / z for p in probs]
            top = sorted(zip(doctors, probs), key=lambda t: -t[1])[:5]
            logprobs = {"content": [{
                "token": best["letter"], "logprob": math.log(max(probs)),
                "top_logprobs": [{"token": d["letter"],
                                  "logprob": math.log(p) if p > 0 else -100.0}
                                 for d, p in top],
            }]}
            return {"text": best["letter"], "logprobs": logprobs,
                    "eval_count": 1, "prompt_eval_count": 60}

        if '"ranking"' in prompt:
            ranking = [doctors[i]["name"] for i in order[:3]]
            obj = {"ranking": ranking,
                   "reason": f"I ranked these because {REASON_PHRASES[_top_driver(best)]}."}
            return {"text": json.dumps(obj), "logprobs": None,
                    "eval_count": 40, "prompt_eval_count": 60}

        reason = f"I chose this doctor because {REASON_PHRASES[_top_driver(best)]}."
        reason_first = prompt.find('"reason"') != -1 and (
            prompt.find('"reason"') < prompt.find('"choice"'))
        if reason_first:
            obj = {"reason": reason, "choice": best["name"]}
        else:
            obj = {"choice": best["name"], "reason": reason}
        return {"text": json.dumps(obj), "logprobs": None,
                "eval_count": 30, "prompt_eval_count": 60}
