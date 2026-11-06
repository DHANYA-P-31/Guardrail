# Module 1 — Labelling Guide v2.0

Matches the spec exactly: two independent softmax heads, emotion and intent.
No separate risk-signal head — crisis severity lives entirely inside the
`crisis_disclosure` intent value.

Write the rules before you collect the data. Every merged public dataset must
be mapped onto *this* schema, and the mapping recorded in `dataset-sources.md`.

## Golden rule

Label what the **text says**, not what you infer about the person. If the label
requires you to imagine facts not in the message, the correct action is
`abstain` / drop the example.

---

## Head 1 — Emotion (single label, softmax)

| Label | Applies when | Not this |
|---|---|---|
| `joy` | happiness, delight, fun | relief (→ `hope_optimism`) |
| `sadness` | grief, loss, loneliness, low mood | hopelessness framed as burden (→ still `sadness`; severity is carried by intent, not a separate emotion) |
| `anger` | frustration, resentment, irritation | — |
| `fear` | anxiety, worry, dread, panic | — |
| `love_care` | affection, concern for someone else, warmth | — |
| `surprise` | unexpectedness, shock (non-threatening) | — |
| `confusion` | uncertainty, not understanding | curiosity without confusion → `neutral` |
| `guilt_remorse` | regret, self-blame, embarrassment | sadness without self-blame |
| `hope_optimism` | relief, gratitude, looking forward | — |
| `neutral` | factual, procedural, no clear affect | — |

## Head 2 — Intent (single label, softmax)

| Label | Test question |
|---|---|
| `emotional_expression` | Are they just narrating how they feel, no explicit ask? |
| `emotional_support` | Are they asking for comfort, presence, reassurance? |
| `advice_seeking` | Are they asking what they should do? |
| `information_seeking` | Are they asking the system to *explain* something factual/clinical? |
| `help_seeking` | Are they asking for concrete action or a referral (for themselves or someone else)? |
| `problem_description` | Are they describing a situation/problem without an explicit ask? |
| `crisis_disclosure` | Do they disclose current/imminent intent to harm themselves? |
| `casual_conversation` | Greetings, thanks, small talk, off-topic? |

**`crisis_disclosure` outranks every other intent.** If it applies, it wins,
regardless of which emotion co-occurs. This is the single most important rule
in this document — Module 2 keys its escalation logic off this one value.

---

## Boundary rules (the ones annotators actually get wrong)

1. **Negation.** "I'm not suicidal, I just feel low" → intent
   `emotional_expression`, NOT `crisis_disclosure`. Do not let keyword
   presence override the sentence's polarity.
2. **Past tense / recovery.** "I used to self-harm, I've been clean a year" →
   `emotional_expression`. Historical ≠ current.
3. **Third party.** "My friend wants to die" → intent `help_seeking` (they're
   asking what to do about someone else), not `crisis_disclosure` — the
   discloser is not the one at risk. Flag this distinction clearly in your
   report; it's a real limitation of a flat intent list with no participant
   field.
4. **Metaphor / hyperbole.** "This deadline is killing me" → `emotional_expression`
   or `casual_conversation`, never `crisis_disclosure`.
5. **Quotation & fiction.** Song lyrics, a hypothetical, "how would I write a
   character who..." → label the *frame* (`information_seeking`), not the
   quoted content.
6. **Ambiguity.** If two annotators could reasonably disagree, mark it
   `ambiguous` and route it to adjudication. Do not guess — this matters most
   for `crisis_disclosure` specifically.
7. **Code-switching.** Tamil–English mixed messages are in scope. Label by
   meaning; flag with `codeswitch: true` so it can be evaluated as its own
   slice.
8. **A single clinical keyword is not enough.** "What are the warning signs
   of suicide?" is `information_seeking`, not `crisis_disclosure` — the word
   "suicide" alone must never be sufficient to trigger the crisis label. (The
   current keyword stub gets this one wrong on purpose, as a demonstration of
   why a trained model is necessary — see `data/adversarial_eval.jsonl`.)

## Annotation process

- Two annotators per item, minimum, on `crisis_disclosure` specifically.
- Report **Cohen's κ per label**. Below 0.6 means the definition is broken,
  not that annotators are careless — rewrite the definition.
- Adjudicate disagreements in a logged session; adjudication notes become the
  next version of this guide.
- Never annotate `crisis_disclosure` cases alone.

## Annotator care

This data is distressing to read in volume. Cap sessions, rotate, and agree
as a team that anyone can stop. Document this in the report — annotator
wellbeing is expected coverage at final-year level.

## Ethics / data handling

- Public research corpora only, under their stated licence and data-use
  agreements. No scraping of individual users.
- No re-identification attempts. Strip usernames, URLs and handles at ingest.
- The trained model is a **router**, not a diagnostic instrument. State this
  on the model card and everywhere metrics are reported.