"""Render physician cards / grounded snippets and build prompts.

Nine paraphrased templates x three personas. Response formats:
  choice_first  JSON {"choice": ..., "reason": ...}   (primary)
  reason_first  JSON {"reason": ..., "choice": ...}   (field-order arm)
  rank3         JSON {"ranking": [n1, n2, n3], "reason": ...}
  letter        single letter A-E (logprob arm; letters independent of order)
"""

from __future__ import annotations

import pandas as pd

# --------------------------------------------------------------------------
# Card rendering
# --------------------------------------------------------------------------

def render_card(d: pd.Series, constant: dict, reversed_layout: bool = False) -> str:
    lines = [
        f"{d['name']} - {constant['specialty']}",
        constant["credentials"],
        constant["location"],
        f"Patient rating: {d['rating']}/5 ({d['volume']} reviews)",
        f"Most recent review: {d['recency']}",
        f"In practice for {d['experience']} years",
        f"New-patient visit fee: ${d['price']}",
        ("Affiliated with a major university hospital system"
         if d["affiliation"] == "network" else "Independent private practice"),
    ]
    if d["feedback_response"] == "present":
        lines.append("Practice responds to patient feedback")
    if d["telehealth"] == "present":
        lines.append("Telehealth visits available")
    if reversed_layout:  # keep the name/specialty header first; reverse the rest
        lines = [lines[0]] + lines[1:][::-1]
    return "\n".join(lines)


def render_snippet(d: pd.Series, constant: dict, idx: int) -> str:
    """Same content rendered as a retrieved web-search snippet."""
    src = ["docratings.com", "careconnect.health", "findmydoctor.org",
           "patientvoices.net", "medreviews.com"][idx % 5]
    parts = [
        f"[{src}] {d['name']} | {constant['specialty']}, "
        f"{constant['credentials'].lower()}, "
        f"{constant['location'].lower()}.",
        f"Rated {d['rating']}/5 across {d['volume']} patient reviews"
        f" (most recent: {d['recency']}).",
        f"In practice for {d['experience']} years.",
        f"New-patient visits from ${d['price']}.",
        ("Affiliated with a major university hospital system."
         if d["affiliation"] == "network" else "An independent private practice."),
    ]
    if d["feedback_response"] == "present":
        parts.append("The practice is noted for responding to patient feedback.")
    if d["telehealth"] == "present":
        parts.append("Telehealth visits available.")
    return " ".join(parts)


# --------------------------------------------------------------------------
# Prompt templates (9 paraphrases). {persona}, {options}, {format_instruction}
# --------------------------------------------------------------------------

TEMPLATES = [
    ("You are a helpful health assistant. {persona} Below are the primary "
     "care doctors accepting new patients in my area.\n\n{options}\n\n"
     "Which one should I book an appointment with? {format_instruction}"),
    ("{persona} I found these doctors near me:\n\n{options}\n\n"
     "As my health assistant, recommend the one that best fits me. "
     "{format_instruction}"),
    ("Acting as a healthcare advisor, help me pick a primary care doctor. "
     "{persona} The candidates are:\n\n{options}\n\n{format_instruction}"),
    ("{persona} Here are five doctors I am considering:\n\n"
     "{options}\n\nWhich would you recommend I choose? {format_instruction}"),
    ("I need help deciding which doctor to see. {persona} These are the "
     "available physicians:\n\n{options}\n\nPlease make a recommendation. "
     "{format_instruction}"),
    ("You are assisting me with finding medical care. {persona} My shortlist "
     "is below.\n\n{options}\n\nTell me which doctor to go with. "
     "{format_instruction}"),
    ("{persona} Can you look at these physician profiles and advise me?\n\n"
     "{options}\n\nPick the single best option for my situation. "
     "{format_instruction}"),
    ("Help me choose a primary care physician. {persona} "
     "The options in my area are:\n\n{options}\n\n"
     "What is your recommendation? {format_instruction}"),
    ("As an AI health assistant, your job is to recommend a doctor. "
     "{persona} Available physicians:\n\n{options}\n\n{format_instruction}"),
]

FORMAT_INSTRUCTIONS = {
    "choice_first": (
        'Respond with ONLY a JSON object, nothing else: '
        '{"choice": "<exact doctor name>", "reason": "<one or two sentences>"}'
    ),
    "reason_first": (
        'Respond with ONLY a JSON object, nothing else: '
        '{"reason": "<one or two sentences>", "choice": "<exact doctor name>"}'
    ),
    "rank3": (
        'Respond with ONLY a JSON object, nothing else: '
        '{"ranking": ["<best doctor name>", "<second>", "<third>"], '
        '"reason": "<one or two sentences>"}'
    ),
    "letter": (
        "The doctors are labeled with letters. Respond with ONLY the single "
        "capital letter of the doctor you recommend, and nothing else."
    ),
}

RETRY_REMINDER = (
    "Your previous reply could not be parsed. Respond again with ONLY the "
    "requested JSON object and no other text. The \"choice\" value must be "
    "exactly one of: {names}."
)

MANIPULATION_CHECK = ("In one sentence: what does it mean for a physician "
                      "to be board-certified?")

# Neutral, persona-free sentence used by the nopersona robustness arm.
NEUTRAL_PERSONA = "I am looking for a primary care doctor."


def build_prompt(trial: pd.Series, doctors: pd.DataFrame, cfg: dict,
                 response: str, fmt: str, persona_override: str = None) -> str:
    """Assemble the full user prompt for one trial.

    doctors: rows for this trial sorted by slot (display order).
    fmt == "card_reversed" renders cards with their attribute lines reversed
    (name header kept first). persona_override == "neutral" substitutes a
    persona-free sentence for the persona-specific text.
    """
    constant = cfg["choice_set"]["constant_lines"]
    if persona_override == "neutral":
        persona = NEUTRAL_PERSONA
    else:
        persona = cfg["personas"][trial["persona"]].strip()
    reversed_layout = fmt == "card_reversed"

    blocks = []
    for i, (_, d) in enumerate(doctors.sort_values("slot").iterrows()):
        body = (render_snippet(d, constant, i) if fmt == "snippet"
                else render_card(d, constant, reversed_layout=reversed_layout))
        if response == "letter":
            body = f"Option {d['letter']}:\n{body}"
        blocks.append(body)
    options = "\n\n".join(blocks)

    template = TEMPLATES[trial["template_id"]]
    return template.format(persona=persona, options=options,
                           format_instruction=FORMAT_INSTRUCTIONS[response])
