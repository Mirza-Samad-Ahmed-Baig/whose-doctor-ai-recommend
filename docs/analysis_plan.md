# Analysis plan — "Whose doctor does the AI recommend?"

Algorithm audit of reputation and demographic signals in LLM-assisted
physician choice. Target journal: **Journal of Medical Internet Research**
(alternates: JAMIA, npj Digital Medicine). Drafted 2026-07-02, to be frozen
on OSF **before** any non-pilot model calls.

## 1. Design summary

Randomized choice-based conjoint, mirroring the hotel audit (research paper
4) so signal weights are comparable across domains. Each trial presents five
synthetic physician cards (family medicine, board-certified, accepting new
patients, near the user — all held constant) and asks the model which doctor
the user should book. Independently randomized per card:

| Attribute | Levels |
|---|---|
| Patient rating | 3.9 / 4.3 / 4.7 (of 5) |
| Number of reviews | 12 / 85 / 400 |
| Most recent review | 3 days ago / 11 months ago |
| Practice responds to feedback | present / absent |
| Affiliation | university hospital system / independent practice |
| New-patient visit fee | $90 / $140 / $190 |
| Telehealth available | present / absent |
| Years in practice | 8 / 18 / 28 |
| **Gender (name-signaled)** | female / male |
| **Ethnicity (name-signaled)** | White / Black / Hispanic / East Asian / South Asian |
| List position | slot 1–5 (randomized display order) |

Names follow correspondence-audit methodology (Bertrand & Mullainathan 2004;
Gaddis 2017): each (gender × ethnicity) cell has 4 first-name exemplars and
4 surnames; analysis is at the cell level, with a name-exemplar placebo test
(within-cell name choice must not matter).

Arms as in the hotel audit: main (3,024 trials), grounded snippets, rank-top-3,
field-order, temp-0, retest (5 reps), letter/logprob, nopersona, card-reversed.
Personas: new-in-town / chronic-condition / budget (uninsured, self-pay).

## 2. Estimands and estimators

Primary: AMCE of each attribute level on P(recommended), linear probability
model with trial-clustered SEs, per model and pooled with model fixed effects
(`src/analysis/amce.py`). Secondary: conditional logit (`clogit.py`);
fee-equivalents via Krinsky–Robb (`tradeoffs.py`), gated on fee-AMCE
monotonicity; stated-vs-revealed reason analysis (`reasons.py`).

## 3. Confirmatory hypotheses (Holm-corrected family, H1–H12)

- **H1–H2**: Higher patient rating (H1) and higher review volume (H2)
  increase recommendation probability.
- **H3**: Recent reviews increase recommendation probability.
- **H4**: Responding to patient feedback increases recommendation probability.
- **H5**: University-hospital affiliation increases recommendation probability.
- **H6**: Higher visit fee decreases recommendation probability.
- **H7**: Telehealth availability increases recommendation probability.
- **H8**: More years in practice increase recommendation probability.
- **H9 (gender parity)**: Name-signaled gender has no effect. Directional
  test two-sided; equivalence assessed by TOST with SESOI ±1.5 pp.
- **H10 (ethnic parity)**: Name-signaled ethnicity has no effect (joint Wald
  over the four non-White dummies vs White reference; per-group TOST at
  ±1.5 pp).
- **H11**: Persona × fee interaction (budget persona weights fee more).
- **H12**: Rating × volume interaction (volume moderates rating diagnosticity).
  *Interpretation caveat (established in pipeline validation):* an LPM
  interaction term can be non-zero under a purely additive logit process
  (probability-scale curvature), so H12 is corroborated only if the
  conditional-logit interaction is also significant with the same sign.
- **H13 (intersectionality)**: Name-signaled gender and ethnicity combine
  *additively*; i.e., there is no gender × ethnicity interaction. Tested as a
  joint Wald test of the female × ethnicity interaction terms. This is the
  null against which we look for intersectional penalties (e.g., a
  Black-female cell worse than gender + ethnicity main effects predict), which
  recent clinical-bias work reports. For presentation we also estimate all ten
  gender × ethnicity cell AMCEs vs the White-male reference from a single
  10-level categorical, each with a TOST equivalence verdict at ±1.5 pp
  (`intersectional.csv`).

### Abstention and hedging (preregistered, descriptive)

Not every non-answer is a failure. We classify each main-arm response as an
**abstention** (the model declines to choose or calls the physicians equally
qualified) and, separately, as **demographic-aware** (the response text refers
to a physician's name, gender, ethnicity, or race). We report both rates per
model (`refusals_summary.csv`). The demographic-aware rate is the
governance-relevant quantity: it measures how often the model's own words
reveal that a name-signaled attribute was salient to its decision. A model
that abstains when only names distinguish the options is behaving
defensibly; one that silently lets names move the choice (H9/H10) while never
mentioning them is the adverse case. This analysis is descriptive
(no confirmatory test) and read alongside H9/H10 and the stated-vs-revealed
comparison.

H9/H10 are **pooled estimands**: the power simulation (`power_mde.csv`) shows
per-model power for small demographic effects is inadequate (~0.6–0.8 at the
assumed sizes), while pooling the 12-model panel gives MDE ≈ 0.6–0.7 pp.
Per-model demographic AMCEs are reported descriptively with CIs, not tested.

Position bias (slot dummies) and the letter-arm decomposition are reported as
in the hotel audit; the position-1 advantage is expressed in $/visit.

## 4. Robustness battery (preregistered)

Template heterogeneity; temperature invariance; grounded-snippet format;
field-order; test–retest; plausibility-cell exclusion; **name-exemplar
placebo** (LR test of first-name dummies given full gender × ethnicity cell
fixed effects); missingness model on parse failures.

## 5. Ethics note

No human subjects; all physician profiles are synthetic. The demographic
manipulation audits the *model*, not people. Findings will be reported at the
group-fairness level with equivalence bounds, avoiding over-claiming from
single-model point estimates.

## 6. Validation

The full pipeline is validated end-to-end by `python -m src.smoke_test`
against a mock data-generating process with known effects, including a known
female (+0.10 logit) and Black-name (−0.15 logit) effect and true-null
effects for the other groups; the suite asserts sign, magnitude, true-null
recovery, gate behavior, and figure/table generation before any real model
is called.
