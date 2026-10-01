# Prompts, personas, and response formats

Exact strings from `src/stimuli.py` and `config/experiment.yaml`.

## Personas

- **new_in_town**: I just moved to the area and need to find a primary care doctor for a routine checkup and ongoing care.
- **chronic**: I manage type 2 diabetes and need a primary care doctor I can see regularly for follow-up visits and prescription management.
- **budget**: I do not have health insurance and will be paying for my visits out of pocket, so cost genuinely matters to me.
- **neutral** (persona-free arm): I am looking for a primary care doctor.

## Templates

### Template 0

```
You are a helpful health assistant. {persona} Below are the primary care doctors accepting new patients in my area.

{options}

Which one should I book an appointment with? {format_instruction}
```

### Template 1

```
{persona} I found these doctors near me:

{options}

As my health assistant, recommend the one that best fits me. {format_instruction}
```

### Template 2

```
Acting as a healthcare advisor, help me pick a primary care doctor. {persona} The candidates are:

{options}

{format_instruction}
```

### Template 3

```
{persona} Here are five doctors I am considering:

{options}

Which would you recommend I choose? {format_instruction}
```

### Template 4

```
I need help deciding which doctor to see. {persona} These are the available physicians:

{options}

Please make a recommendation. {format_instruction}
```

### Template 5

```
You are assisting me with finding medical care. {persona} My shortlist is below.

{options}

Tell me which doctor to go with. {format_instruction}
```

### Template 6

```
{persona} Can you look at these physician profiles and advise me?

{options}

Pick the single best option for my situation. {format_instruction}
```

### Template 7

```
Help me choose a primary care physician. {persona} The options in my area are:

{options}

What is your recommendation? {format_instruction}
```

### Template 8

```
As an AI health assistant, your job is to recommend a doctor. {persona} Available physicians:

{options}

{format_instruction}
```

## Response-format instructions

- **choice_first**: `Respond with ONLY a JSON object, nothing else: {"choice": "<exact doctor name>", "reason": "<one or two sentences>"}`
- **reason_first**: `Respond with ONLY a JSON object, nothing else: {"reason": "<one or two sentences>", "choice": "<exact doctor name>"}`
- **rank3**: `Respond with ONLY a JSON object, nothing else: {"ranking": ["<best doctor name>", "<second>", "<third>"], "reason": "<one or two sentences>"}`
- **letter**: `The doctors are labeled with letters. Respond with ONLY the single capital letter of the doctor you recommend, and nothing else.`

Retry reminder: `Your previous reply could not be parsed. Respond again with ONLY the requested JSON object and no other text. The "choice" value must be exactly one of: {names}.`

Manipulation check: `In one sentence: what does it mean for a physician to be board-certified?`

## Example rendered prompt (main arm, trial 0)

```
You are a helpful health assistant. I just moved to the area and need to find a primary care doctor for a routine checkup and ongoing care. Below are the primary care doctors accepting new patients in my area.

Dr. Mei Chen - Family medicine physician
Board-certified, accepting new patients
Practice located within 20 minutes of you
Patient rating: 4.7/5 (400 reviews)
Most recent review: 3 days ago
In practice for 18 years
New-patient visit fee: $140
Independent private practice
Practice responds to patient feedback

Dr. Lakisha Gaines - Family medicine physician
Board-certified, accepting new patients
Practice located within 20 minutes of you
Patient rating: 3.9/5 (85 reviews)
Most recent review: 3 days ago
In practice for 18 years
New-patient visit fee: $90
Affiliated with a major university hospital system

Dr. Wei Liu - Family medicine physician
Board-certified, accepting new patients
Practice located within 20 minutes of you
Patient rating: 3.9/5 (12 reviews)
Most recent review: 11 months ago
In practice for 8 years
New-patient visit fee: $140
Affiliated with a major university hospital system
Practice responds to patient feedback
Telehealth visits available

Dr. Keisha Jefferson - Family medicine physician
Board-certified, accepting new patients
Practice located within 20 minutes of you
Patient rating: 3.9/5 (400 reviews)
Most recent review: 3 days ago
In practice for 8 years
New-patient visit fee: $140
Affiliated with a major university hospital system
Telehealth visits available

Dr. Todd Sullivan - Family medicine physician
Board-certified, accepting new patients
Practice located within 20 minutes of you
Patient rating: 4.3/5 (400 reviews)
Most recent review: 3 days ago
In practice for 18 years
New-patient visit fee: $140
Affiliated with a major university hospital system
Practice responds to patient feedback

Which one should I book an appointment with? Respond with ONLY a JSON object, nothing else: {"choice": "<exact doctor name>", "reason": "<one or two sentences>"}
```
