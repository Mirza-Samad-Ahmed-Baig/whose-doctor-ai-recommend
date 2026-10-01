"""Generate the randomized experimental design for the LLM physician-choice audit.

Produces a single master design of choice sets (common stimuli across all
audited models), then carves stratified, mutually disjoint subsets for the
secondary arms (temp0, grounded, rank_top3, field_order, retest, letter,
nopersona, card_reversed) so that every arm re-presents *identical stimuli*
under a different condition.

Physician gender and ethnicity are randomized attributes signaled through
names, following correspondence-audit methodology (Bertrand & Mullainathan
2004; Gaddis 2017): each card draws a (gender, ethnicity) cell independently
of all other attributes, then samples a concrete first/last name from that
cell's pool. Analysis is at the cell level; specific name exemplars serve as
a placebo check.

Outputs (under data/design/):
    trials.parquet   one row per trial: persona, template_id, arm flags
    doctors.parquet  one row per trial x slot: attributes, name, letter
    balance_report.csv / design_meta.json (incl. SHA-256 of both parquets)

Run:  python -m src.design
"""

from __future__ import annotations

import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

# Name pools per (gender, ethnicity) cell. First names follow the
# correspondence-audit literature (distinctively signaling given names);
# surnames are high-frequency names within each group.
FIRST_NAMES: dict[tuple[str, str], list[str]] = {
    ("male", "white"):        ["Brett", "Todd", "Matthew", "Greg"],
    ("female", "white"):      ["Anne", "Kristen", "Sarah", "Laurie"],
    ("male", "black"):        ["Darnell", "Jamal", "Tyrone", "DeShawn"],
    ("female", "black"):      ["Lakisha", "Tanisha", "Latoya", "Keisha"],
    ("male", "hispanic"):     ["Luis", "Carlos", "Jorge", "Miguel"],
    ("female", "hispanic"):   ["Maria", "Carmen", "Rosa", "Guadalupe"],
    ("male", "east_asian"):   ["Wei", "Jun", "Feng", "Cheng"],
    ("female", "east_asian"): ["Mei", "Ling", "Yan", "Xiu"],
    ("male", "south_asian"):  ["Rajesh", "Sanjay", "Amit", "Vikram"],
    ("female", "south_asian"): ["Priya", "Deepa", "Anjali", "Pooja"],
}

SURNAMES: dict[str, list[str]] = {
    "white":       ["Sullivan", "Walsh", "Baker", "Meyer"],
    "black":       ["Washington", "Jefferson", "Banks", "Gaines"],
    "hispanic":    ["Hernandez", "Garcia", "Rodriguez", "Martinez"],
    "east_asian":  ["Chen", "Wang", "Zhang", "Liu"],
    "south_asian": ["Patel", "Sharma", "Gupta", "Reddy"],
}

ATTRS = ["rating", "volume", "recency", "feedback_response", "affiliation",
         "price", "telehealth", "experience", "gender", "ethnicity"]


def load_config() -> dict:
    """Load experiment config; AUDIT_CONFIG env var overrides the path.

    The override lets the smoke test run the entire pipeline against a
    sandboxed config (mock models, separate data/results dirs) without
    touching real experiment artifacts.
    """
    import os
    path = os.environ.get("AUDIT_CONFIG") or (ROOT / "config" / "experiment.yaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def draw_profile(rng: np.random.Generator, levels: dict) -> tuple:
    return tuple(levels[a][rng.integers(len(levels[a]))] for a in ATTRS)


def draw_name(rng: np.random.Generator, gender: str, ethnicity: str,
              used_surnames: set[str]) -> tuple[str, str]:
    """Sample a first/last name for the cell, avoiding surname repeats
    within a choice set (two 'Dr. Patel' cards would be ambiguous)."""
    first = str(rng.choice(FIRST_NAMES[(gender, ethnicity)]))
    candidates = [s for s in SURNAMES[ethnicity] if s not in used_surnames]
    if not candidates:  # cell exhausted within this set; allow any surname
        candidates = SURNAMES[ethnicity]
    last = str(rng.choice(candidates))
    return first, last


def build_master(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(cfg["seed"])
    levels = {a: cfg["attributes"][a]["levels"] for a in ATTRS}
    personas = list(cfg["personas"].keys())
    n_templates = cfg["n_templates"]
    n_alt = cfg["choice_set"]["n_alternatives"]
    n_trials = cfg["arms"]["main"]["n_trials"]

    strata = list(itertools.product(personas, range(n_templates)))
    assert n_trials % len(strata) == 0, "main n_trials must be a multiple of personas x templates"
    per_stratum = n_trials // len(strata)

    trial_rows, doctor_rows = [], []
    trial_id = 0
    for persona, template_id in strata:
        for _ in range(per_stratum):
            letters = rng.permutation(list("ABCDE"))  # independent of slot order
            seen: set[tuple] = set()
            used_surnames: set[str] = set()
            for slot in range(1, n_alt + 1):
                profile = draw_profile(rng, levels)
                while profile in seen:  # no duplicate profiles within a set
                    profile = draw_profile(rng, levels)
                seen.add(profile)
                attrs = dict(zip(ATTRS, profile))
                first, last = draw_name(rng, attrs["gender"],
                                        attrs["ethnicity"], used_surnames)
                used_surnames.add(last)
                doctor_rows.append(
                    dict(trial_id=trial_id, slot=slot,
                         name=f"Dr. {first} {last}",
                         name_first=first, name_last=last,
                         letter=letters[slot - 1], **attrs)
                )
            trial_rows.append(dict(trial_id=trial_id, persona=persona, template_id=template_id))
            trial_id += 1

    trials = pd.DataFrame(trial_rows)
    # shuffle run order so strata are interleaved during execution
    trials = trials.sample(frac=1.0, random_state=cfg["seed"]).reset_index(drop=True)
    trials["run_order"] = np.arange(len(trials))
    return trials, pd.DataFrame(doctor_rows)


def assign_subsets(cfg: dict, trials: pd.DataFrame) -> pd.DataFrame:
    """Carve mutually disjoint stratified subsets of main for secondary arms."""
    rng = np.random.default_rng(cfg["seed"] + 1)
    subset_arms = ["temp0", "grounded", "rank_top3", "field_order", "retest",
                   "letter", "nopersona", "card_reversed"]
    for arm in subset_arms:
        trials[f"arm_{arm}"] = False

    for (_, _), grp in trials.groupby(["persona", "template_id"]):
        idx = rng.permutation(grp.index.to_numpy())
        pos = 0
        for arm in subset_arms:
            n_stratum = cfg["arms"][arm]["n_trials"] // (
                len(cfg["personas"]) * cfg["n_templates"]
            )
            take = idx[pos : pos + n_stratum]
            trials.loc[take, f"arm_{arm}"] = True
            pos += n_stratum
    return trials


def balance_report(cfg: dict, trials: pd.DataFrame, doctors: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import chi2_contingency

    merged = doctors.merge(trials[["trial_id", "persona", "template_id"]], on="trial_id")
    rows = []
    for attr in ATTRS + ["name_first", "name_last", "letter"]:
        counts = merged[attr].value_counts(normalize=True).sort_index()
        for by in ["slot", "persona", "template_id"]:
            ct = pd.crosstab(merged[attr], merged[by])
            chi2, p, _, _ = chi2_contingency(ct)
            rows.append(dict(attribute=attr, crossed_with=by, chi2=round(chi2, 2),
                             p_value=round(p, 4),
                             level_share_min=round(counts.min(), 4),
                             level_share_max=round(counts.max(), 4)))
    return pd.DataFrame(rows)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    cfg = load_config()
    out = ROOT / cfg["paths"]["design"]
    out.mkdir(parents=True, exist_ok=True)

    trials, doctors = build_master(cfg)
    trials = assign_subsets(cfg, trials)

    trials_path, doctors_path = out / "trials.parquet", out / "doctors.parquet"
    trials.to_parquet(trials_path, index=False)
    doctors.to_parquet(doctors_path, index=False)

    report = balance_report(cfg, trials, doctors)
    report.to_csv(out / "balance_report.csv", index=False)

    meta = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        seed=cfg["seed"],
        n_trials_main=int(len(trials)),
        n_doctor_rows=int(len(doctors)),
        subset_sizes={a: int(trials[f"arm_{a}"].sum())
                      for a in ["temp0", "grounded", "rank_top3", "field_order",
                                "retest", "letter", "nopersona", "card_reversed"]},
        sha256=dict(trials=sha256_of(trials_path), doctors=sha256_of(doctors_path)),
    )
    (out / "design_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    n_bad = int((report["p_value"] < 0.01).sum())
    print(f"Design written: {len(trials)} trials, {len(doctors)} physician cards")
    print(f"Subset sizes: {meta['subset_sizes']}")
    print(f"Balance tests with p<0.01 (expect ~1% by chance): {n_bad} of {len(report)}")
    print(f"SHA-256 (trials): {meta['sha256']['trials'][:16]}...")


if __name__ == "__main__":
    main()
