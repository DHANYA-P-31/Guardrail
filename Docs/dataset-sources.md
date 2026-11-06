# Dataset Sources & Label Mapping

Fill the **Status** and **Mapping** columns as you go. This table is a report
section, not admin overhead — the mapping decisions are a genuine contribution.

## Candidates

| Source | Covers | Access | Status | Notes |
|---|---|---|---|---|
| GoEmotions (Google) | 27 emotions, Reddit | Open | ✅ | Downloaded as goemotions_1/2/3.csv; maps many→our 10 |
| dair-ai/emotion | 6 emotions, Twitter | Open (HF) | ✅ | Downloaded as dair-ai-{train,validation,test}.parquet |
| Dreaddit | stress vs non-stress, Reddit | Open | ✅ | Downloaded as dreaddit.csv; mapped to `fear`/`neutral` |
| Counsel-Chat | counsellor Q&A (31 topics) | Open | ✅ | Downloaded; `topic` + `questionTitle` pattern rules → 4 intent classes |
| MHDialog | mental-health support dialogues, intent + risk severity | Open (HF) | ✅ | Downloaded as MHDialog-{train,validation,test}.parquet |
| DailyDialog | general chit-chat, dialogue acts + emotion, per-utterance | Open | ✅ | Downloaded as dailydialog{train,test,validation}.csv |
| CLPsych shared tasks | risk severity | **Application + DUA** | ☐ | Fallback only if `crisis_disclosure` support from MHDialog is still too thin |
| UMD Reddit Suicidality | 4-level risk | **Application + DUA** | ☐ | Same, fallback only |
| Team-written eval set | all labels | Ours | ✅ | See `data/adversarial_eval.jsonl` (15 cases) |

**`emotion_reliable` / `intent_reliable` flags.** Most sources only have real
per-example annotation for ONE of the two heads — dair-ai/GoEmotions/Dreaddit
have real emotion but no real intent; Counsel-Chat/MHDialog have real intent
but no real emotion. `build_dataset.py` tags every row with both flags;
`baseline.py` and `train.py` train each head only on rows where that head's
flag is `True`, rather than letting a blanket per-source default (e.g. every
dair-ai row defaulting to `emotional_expression`) masquerade as a real
annotation. DailyDialog is the one source with real signal for BOTH heads,
per utterance — but even there, `directive`/`commissive` dialogue acts are
marked `intent_reliable=False` (they describe conversational structure, not
support-seeking behaviour, and forcing them onto the intent taxonomy would
repeat the exact mistake first caught in Counsel-Chat).

**`crisis_disclosure` status — RESOLVED, with an accepted limitation, not an
open question.** Counsel-Chat's `topic` field has `self-harm` (1 row) and no
`suicide` category — negligible. MHDialog is the only real source:
`Dialog Intent == "CrisisEscalation"` maps directly, and its `Level` field
(confirmed 6-value risk-severity scale: `Not Related` 398, `Moderate` 101,
`Minor` 84, `Unsure` 70, `Severe` 29, `No` 18) overrides a lower-severity
intent label when risk is high enough. `Severe` → crisis; `No` is mapped to
non-crisis as an unverified best guess (flagged in code, not confirmed
against the original annotation scheme). An ordering bug — Dialog Intent
was checked before Level, so a severity override could never rescue a row
whose Dialog Intent fell in the dropped {Recovery, Avoidance, Adversarial}
set — was found and fixed; this alone took the yield from 11 to 27 examples
on the real data.

**Final count: 27 training examples for `crisis_disclosure`**, against
26,540 for the largest intent class. **Decision: proceed with this as a
documented limitation** rather than pursue CLPsych/UMD or further
hand-labelling, given project timeline — see the Known Limitation section
in the top-level README for the exact framing to use in the report.
`scripts/train.py` and `scripts/baseline.py` both class-weight their losses
to stop this imbalance from drowning the signal entirely, but weighting
does not create information that isn't there; expect this class's recall
to be noisy. If priorities change, CLPsych/UMD DUA applications are the
documented next step (see their rows above).

Do not use a dataset whose licence you have not read. A licence violation in a
final-year report is worse than a lower F1.

## Mapping rule

For each source, write an explicit `source_label → our_label` table. Where a
source label has no clean target, **drop it** rather than forcing a mapping.
Record every drop with a reason.

Example shape:

```
GoEmotions:
  nervousness, fear      -> anxiety
  sadness, grief         -> sadness
  disappointment         -> DROPPED (ambiguous between sadness and neutral)
```

## Limitations to state explicitly in the report

1. **Domain shift.** These are public forum posts. Our deployment target is
   helpline conversation. Vocabulary, length and disclosure style differ.
   Quantify it if you can: report metrics separately on the team-written
   conversational eval set.
2. **Label incompatibility.** Sources were annotated under different guides by
   different populations. Merging them injects label noise that caps achievable
   agreement. Report inter-source agreement on any overlapping examples.
3. **Selection bias.** People who post publicly about distress are not
   representative of people who call a helpline.
4. **English-dominant.** Code-switched Tamil–English is underrepresented in
   every source above. This is a known gap; report it, and evaluate the slice
   separately rather than letting it vanish into the average.
5. **Temporal drift.** Language around mental health changes. Record the
   collection date of every source.

## Splits

- Stratify on `emotion` and `intent` (two heads only — no risk-signal split
  needed since that head was dropped from the schema, matching the spec).
- **Group split** by author/thread where the source provides it, otherwise near-
  duplicate posts leak between train and test and inflate every number.
- Freeze the test set once. Write its hash into the model card. Never look at it
  until the final run.
- Watch `crisis_disclosure` support specifically once the merge runs —
  `build_dataset.py` will print a warning if it's too rare to stratify on,
  and that warning is the signal to either pursue CLPsych/UMD or fold the
  team-written eval set's crisis cases into training data as a stopgap.