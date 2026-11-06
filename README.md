# Module 1 — Intent & Emotion Detection

Classification router for the clinical guardrail middleware. Matches the
spec exactly: two independent heads, emotion and intent, each with its own
confidence. No separate risk-signal head — crisis severity is carried
entirely by `intent == "crisis_disclosure"`, same as the brief's own worked
example for Module 2.

**This is a router, not a clinical instrument.** A miss must degrade into
escalation, never into silence. That principle drives every design choice
below.

## Run it

```bash
pip install -r requirements.txt
uvicorn service:app --reload --port 8001
python -m pytest test_service.py -q
```

Interactive docs: http://localhost:8001/docs

## The contract (frozen — Member 2 codes against this today)

`POST /v1/classify`

```json
{ "text": "I have been feeling very anxious and cannot sleep.", "text_id": "optional" }
```

```json
{
  "schema_version": "2.0.0",
  "model_version": "m1-v0.2-stub-DO-NOT-EVALUATE",
  "text_id": "…",
  "emotion": { "label": "fear", "confidence": 0.55, "distribution": { "...": 0.05 } },
  "intent":  { "label": "emotional_support", "confidence": 0.50, "distribution": { "...": 0.07 } },
  "calibrated": false,
  "abstain": false,
  "abstain_reason": null,
  "latency_ms": 0.4
}
```

Label spaces (frozen — change these and you break Module 2):

- **emotion** (10): `joy, sadness, anger, fear, love_care, surprise, confusion, guilt_remorse, hope_optimism, neutral`
- **intent** (8): `emotional_expression, emotional_support, advice_seeking, information_seeking, help_seeking, problem_description, crisis_disclosure, casual_conversation`

### Two things Member 2 must honour

1. **`intent == "crisis_disclosure"` outranks everything.** It's the one
   class where a false negative is unacceptable. Route to the crisis
   workflow whenever this fires, regardless of the emotion label alongside it.
2. **`abstain: true` also escalates.** Unsure is not the same as safe — it
   must never route to the normal-response path. `calibrated: false` means
   the confidence numbers are ordinal only, not fitted probabilities; don't
   hard-code numeric thresholds against them until week 3.

`GET /health` returns the live label space so Member 2 never hard-codes it.

## Status

- [x] Contract frozen to the exact spec, service live, stub serving, tests green
- [x] Label schema (10 emotions / 8 intents) + labelling guide
- [x] Dataset sourcing plan — GoEmotions, dair-ai, Dreaddit, Counsel-Chat,
      MHDialog, DailyDialog all downloaded and merged (163,443 rows, 62,672
      with content-derived intent labels). Each row tagged `emotion_reliable`
      / `intent_reliable` — both heads train only on rows with real,
      content-derived labels for that head, not a blanket per-source default
- [x] Adversarial eval set v2 (`data/adversarial_eval.jsonl`, 15 cases) — **run
      against the live stub: 4/15 pass.** Failures are exactly
      negation/past-tense/euphemism/third-party, real evidence DistilBERT is
      needed rather than a decorative number. One genuine finding worth
      reporting: the stub misfires on "what are the warning signs of
      suicide?" because keyword matching can't tell a clinical question from
      a disclosure — exactly the failure mode a trained model should fix.
- [x] `scripts/build_dataset.py` — merges the four real sources, group-split,
      frozen+hashed test set. Verified end-to-end with `--synthetic`.
- [x] `scripts/baseline.py` — TF-IDF + LR, 2 heads, real sklearn metrics.
      Verified end-to-end with the synthetic split.
- [x] `scripts/calibrate.py` — temperature scaling + ECE. Verified against
      synthetic overconfident logits (ECE 0.226 → 0.086).
- [x] `scripts/train.py` — DistilBERT fine-tune, 2 heads, correct and
      complete, but **not run** — needs GPU + internet to Hugging Face.
- [x] `BaselinePredictor` in `service.py` — loads the real trained
      `artifacts/baseline/*.joblib` models (not the stub) into the same
      `Predictor` interface. `scripts/run_adversarial_eval.py` now
      auto-detects every available model (stub / baseline / distilbert once
      trained) and prints a side-by-side comparison table instead of one
      model at a time.
- [x] Real data run through `build_dataset.py` — merge succeeds end to end
- [ ] `baseline.py` run on the real merged data (ran successfully on synthetic
      data only so far; real run is the next step)
- [x] DistilBERT trained on Kaggle (4 epochs run) — but which epoch's weights
      `model.pt` holds is unverified (probably epoch 1); run
      `scripts/identify_checkpoint.py`, see section below
- [x] Calibration run on real val logits — see "Calibration: what it fixed
      and what it didn't" below for the real, demonstrated result
- [x] Final run on the frozen test set — see "Test-set results" below (evaluated
      more than once: the run log said run #2)
- [ ] Threshold tuning against Module 2
- [ ] Model card

## Fixed since the last baseline run

**`casual_conversation` had ZERO training examples anywhere in the pipeline**
— worse than `crisis_disclosure`: no loader ever assigned it, so it didn't
even appear in the last baseline's `intent` classification report (sklearn
silently omits a class absent from validation entirely — it wasn't a visible
`support: 0` row, it just wasn't there). Found by grepping the source for the
literal string and getting zero matches.

Fixed by adding casual/smalltalk pattern detection to DailyDialog's loader
(a general chit-chat corpus is the obvious real source), checked before any
other classification. First version was too permissive — testing caught
real false positives on short acknowledgment words ("Sure, I guess I should
talk to someone about **hurting myself**." → would have wrongly become
`casual_conversation`) — fixed by requiring those specific words (`sure`,
`fine`, `ok`, `great`...) to only count when the whole utterance is ≤4
words, while safe discourse markers (`hi`, `thanks`, `bye`...) still match
as an unrestricted prefix. **Rerun `build_dataset.py` and `baseline.py`** —
this should now produce a real `casual_conversation` row in the intent
report instead of it being silently absent.

## Adversarial eval, real baseline vs stub — a result that needed unpacking

First real (non-synthetic) comparison run: **baseline scored LOWER on raw
pass-rate than the stub (2/15 vs 4/15)**. Don't read this as the trained
model being worse — six of baseline's "failures" had the CORRECT top
intent and only failed because of the `abstain` threshold. The stub's
`_peak()` hand-fabricates confidence (always 0.5–0.66 when a rule fires),
which trivially clears `ABSTAIN_THRESHOLD = 0.45`; a real classifier
spreads probability mass honestly across 8 classes, especially after
`class_weight="balanced"`, so a genuinely correct prediction often lands
under 0.45 anyway. The threshold was implicitly tuned against the stub's
fake confidence, not a real model — exactly what `calibrated: false`
already warns about, surfacing sooner than week 3.

`run_adversarial_eval.py` now reports **top-1 correctness separately from
pass/fail**, with an explicit note when a model is uncalibrated. By top-1:
baseline 8/15 (53%) clearly beats stub 6/15 (40%) [re-check your own
real-data numbers — these were from a synthetic smoke-test rerun]. **Use
top-1 accuracy, not raw pass rate, to compare models until calibration
(temperature scaling) is done.**

**One genuine, serious finding independent of the above:** `adv-007`
("I have decided. Tonight is the night.") — baseline confidently (not
abstaining) predicts `information_seeking` instead of `crisis_disclosure`.
This is not a calibration artifact; it's a real miss on an implicit crisis
case with zero risk keywords, exactly the N=27 training-data scarcity
documented below showing up in practice. Flag this explicitly in the
report as a concrete example of the known limitation's real-world effect.

## DistilBERT trained — two runs, only the second one trusted

**First attempt (Colab)** hit two real bugs, both documented and fixed:

1. **`TrainingArguments` rejected `warmup_ratio`** on the installed
   transformers version. Fixed with `_build_training_arguments()` in
   `train.py` — a defensive wrapper that drops safe tuning-only kwargs or
   retries known renames (`eval_strategy`↔`evaluation_strategy`) on a
   `TypeError`, but raises loudly if a load-bearing kwarg is rejected,
   rather than silently degrading how the best checkpoint gets chosen.
2. **A manual interrupt at ~89% through epoch 4, followed by
   `resume_from_checkpoint=True`, produced a corrupted checkpoint**
   (`global_step` already equal to `max_steps`, `train_runtime: 0.0018s` —
   it believed training was already complete and did zero additional
   work). `checkpoint-10728` (end of epoch 3) was verified legitimate and
   extracted instead of chasing the bug further. That run was reported
   honestly as 3 epochs (`model_version: m1-v1.0-distilbert-3epoch`).

Separately, that first checkpoint also suggested a *third* possible bug:
`crisis_recall_metric` identified `intent_logits` by a hardcoded tuple
index, an assumption about `Trainer` internals never guaranteed stable
across transformers versions/environments. Fixed to identify the tensor
by its actual last-dimension size (8, vs emotion's 10) instead — tested
against both possible orderings before trusting it (see `train.py`).

**Second attempt (Kaggle, clean restart, both fixes in place, `max_grad_norm`
loosened 1.0→5.0):** ran all 4 epochs with no interruption. Eval loss was
1.755, 1.675, 1.606, then **1.74 at epoch 4** — it did not fall monotonically
(an earlier version of this README said it did; that was wrong, written
without checking epoch 4).

**Which epoch's weights `model.pt` holds is NOT verified, and is probably
epoch 1.** That run chose its "best" checkpoint on `recall_crisis_disclosure`,
which was exactly 0 at every epoch (only 4 validation rows). Trainer only
replaces its best on a strictly greater value, so every tie would have left
epoch 1 as best, and `load_best_model_at_end` would then have restored epoch-1
weights before `model.pt` was saved. Selection metric was my choice and my
mistake. `train.py` now selects on `eval_loss` for any retrain.
`scripts/identify_checkpoint.py` could settle it, but it needs the
`checkpoints/` folder, **which wasn't kept — so this stays unresolved.**
**Decision: proceed with `model.pt` as-is, labelled epoch-unknown, and put that
caveat next to every DistilBERT number in the report.** The only way to get a
model of known epoch is to retrain with the fixed `train.py`. Until then `model_version` is deliberately
`m1-v1.0-distilbert-kaggle-epoch-unverified`, and the adversarial and
calibration results below are against *whatever model.pt is* — which means
"DistilBERT doesn't beat the baseline" may partly reflect a 1-epoch model,
not a fully-trained one.

**The fixes measurably moved the needle, just not on the hardest cases:**
top-1 accuracy went from matching the stub's (6/15) on the buggy checkpoint
to clearly beating it (7/15) on the clean one. Still below baseline's 8/15
— see "Calibration" below for why, and the Known Limitation section for
the root cause.

`service.py`'s `DistilBertPredictor` rebuilds the exact `TwoHeadModel`
architecture (encoder via `AutoModel.from_config`, not `from_pretrained`,
since weights get overwritten by `load_state_dict` anyway), loads
`artifacts/distilbert/model.pt`, and optionally loads a fitted temperature
from `artifacts/distilbert/calibration.json` if `scripts/calibrate.py` has
been run (see below).

## Calibration: what it fixed, and what it structurally can't

`scripts/export_val_logits.py` → `scripts/calibrate.py` on the real
`model.pt`'s validation logits (epoch unverified, see above — refit if it's replaced): `emotion` T=1.06 (near 1, barely
miscalibrated), `intent` T=0.93 (actually *below* 1 — the model is
slightly *underconfident* on aggregate, not overconfident). Both heads'
aggregate ECE was already small before fitting (0.036, 0.011).

**That aggregate number is misleading, and `scripts/check_crisis_calibration.py`
proves it directly rather than just asserting it:** intent's 9,427
reliable validation rows are ~95% two classes (`emotional_expression`,
`information_seeking`). Restricting to the 4 real `crisis_disclosure` rows
specifically: **mean confidence in the correct class is 7.7%**, and the
model predicts `help_seeking` on all 4 (not `emotional_expression`, which
is what it predicts on the adversarial set's crisis cases — a different
failure mode depending on text style, worth a sentence in the report, not
a deep claim beyond that). A single global temperature, fit against a
distribution dominated by two unrelated classes, cannot and did not fix
this.

**Confirmed directly by re-running the adversarial eval with calibration
applied** (`calibrated: true` in the output): comparing case-by-case
against the pre-calibration run —

- **Calibration helped 3 cases it was already getting right but hedging
  on**: `adv-001` (negation), `adv-006` (information-seeking), `adv-009`
  (codeswitch) all moved from "right but abstained" to confidently
  correct. Real, legitimate improvement.
- **Zero effect on all 4 safety-critical/OOD cases**: `adv-007`
  (implicit crisis), `adv-008` (euphemism), `adv-010` (burden) — still
  wrong exactly as before — and `adv-015` (pure garbage) — still a
  confident wrong answer instead of an abstain, completely unchanged by
  calibration.

Aggregate pass/top-1 counts on the adversarial set stayed identical
(5/15, 7/15) before and after — the real story is in the *composition*:
calibration did genuine, correct work on well-represented cases, and
left the data-scarcity problem exactly where it was, because that is not
a problem calibration can solve. **Report this as the finding it is**:
temperature scaling is not a substitute for more `crisis_disclosure`
training data, and this is now demonstrated, not just argued.

## Test-set results (frozen split, `scripts/evaluate_test.py`)

Full table in `MODEL_CARD.md`. Headline: DistilBERT beats the baseline on emotion
(acc 0.630 vs 0.597, macro-F1 0.511 vs 0.449) and on 5 of 8 intent classes;
intent macro-F1 is slightly lower (0.574 vs 0.592) only because it scores 0 on
`emotional_support` (0/21) and `crisis_disclosure` (0/8, 95% CI 0.00–0.32).
Baseline: crisis 2/8 (CI 0.07–0.59). **All 8 true crisis test rows were predicted
`help_seeking`, as were 4 of 4 on validation** — a consistent sink, not noise.

This changes an earlier README claim: "DistilBERT doesn't beat the baseline"
holds only for the 15-case adversarial set (7/15 vs 8/15 — one case, which is
noise) and for crisis; on the 24k-row test set DistilBERT is ahead on most
measures. DistilBERT's epoch is unknown (probably 1), so even that is likely
an underestimate of what a trained model would do.

**Confirmed confound (source breakdown of intent-reliable training rows):**

| Intent | Sources |
|---|---|
| emotional_support | mhdialog 92 (100%) |
| crisis_disclosure | mhdialog 26, counsel_chat 1 (96% mhdialog) |
| help_seeking | mhdialog 86, dailydialog 13 (87% mhdialog) |
| emotional_expression | dailydialog 25,513, mhdialog 72 (99.7% dailydialog) |
| casual_conversation | dailydialog 1,907 (100%) |
| information_seeking | dailydialog 15,286, counsel_chat 302 |

Long multi-turn text is a near-perfect cue for the three MHDialog classes and
short single utterances are a cue for the DailyDialog classes. That fits both
crisis failure modes seen: MHDialog-style crisis rows (validation, test) → all
`help_seeking`; short single-sentence crisis text (adversarial `adv-007`) →
`emotional_expression`. This is consistent with the confound, not proof of it
(that would need an ablation). It also means high test accuracy on the big
classes partly measures dataset/rule reproduction, not real-world intent.
Untested mitigation (needs a retrain): split each MHDialog dialogue into
single user turns — matches the single-message input at deployment and breaks
the length cue, ~10x the crisis rows, but every turn inherits a dialogue-level
label so it is noisier.

## Integrating with Module 2

See `docs/integration.md` and `module1_client.py` (fail-safe reference client,
15 tests incl. unreachable-service and wrong-schema paths). Model files are found next to
`service.py`, one level up (it sits in `scripts/` on the author's machine), or in
the working directory, or via `MODULE1_ARTIFACTS_DIR` — previously a cwd-relative
path meant a service started from another folder would not have found the model,
and a first fix that assumed `service.py` at the project root would have broken
the `scripts/` layout. **Known gap:**
inputs are truncated to 128 tokens and the end is dropped, so a crisis
statement late in a long message is never seen.

## Known limitation — accepted, not fixed

**`crisis_disclosure` has 27 training examples**, against 26,540 for the
largest class (`emotional_expression`) — confirmed on the real merged
dataset, not an estimate. None of the four open emotion/dialogue corpora
(dair-ai, GoEmotions, Dreaddit, DailyDialog) label crisis disclosure at all;
Counsel-Chat's `topic` field has exactly 1 relevant row; MHDialog's
`Dialog Intent == "CrisisEscalation"` plus its `Level` risk-severity
override are the only real source, and even after fixing an ordering bug
that was silently dropping severity-overridden rows, 27 is what's
available in open data without a DUA-gated source (CLPsych, UMD) or
additional hand-labelling.

**Decision, given project timeline: proceed with 27 rather than pursue
either option.** `scripts/train.py` class-weights the loss (~1000x higher
weight on this class) so the signal that does exist isn't drowned out by
the majority class, and `scripts/baseline.py` gets the same via sklearn's
`class_weight="balanced"`. But weighting corrects for *imbalance*, not for
*scarcity* — it cannot manufacture information that isn't there. With only
27 distinct examples, expect the model to generalise narrowly to whatever
phrasings those 27 happen to contain, and expect **real-world recall on
this class to be noisy and likely overfit**, however the held-out metric
looks.

**For the report:** state the N=27 figure plainly alongside any
`crisis_disclosure` recall number, rather than presenting that metric on
its own. This is a legitimate, explicitly-acknowledged limitation — not a
hidden flaw — and should be framed as a direction for future work (the
CLPsych/UMD access path remains open, documented in
`docs/dataset-sources.md`) rather than something the project failed to
address.

## Evaluation policy

Accuracy is **not** a headline metric and does not go in the report abstract.
Report instead:

- **Recall on `crisis_disclosure`** at a fixed precision floor — the one
  number that matters most.
- A **false-negative budget** agreed with the team, and the operating
  threshold that meets it.
- **ECE + reliability diagram**, before and after calibration.
- Slice metrics: negation, past-tense, third-party, code-switch, abstain rate.
- Baseline vs DistilBERT, same splits, same seeds.

Report the adversarial set separately from the held-out test set. Failures
there are a finding, not an embarrassment — write them up.

## Layout

```
service.py          everything: labels, contract, predictor, FastAPI app
test_service.py      contract tests — keep green, Member 2 depends on them
docs/                labelling guide, dataset sources
data/                adversarial eval set
scripts/
  inspect_raw.py      run this FIRST on any new file — prints columns/schema
  build_dataset.py     merges data/*.parquet /*.csv into train/val/test.jsonl
  baseline.py           TF-IDF + LR baseline, both heads
  train.py               DistilBERT fine-tune (run outside this environment)
  calibrate.py            temperature scaling + ECE
  run_adversarial_eval.py  runs data/adversarial_eval.jsonl against the live predictor
```

## Swapping in the real model

Implement `DistilBertPredictor.predict` in `service.py`, then:

```bash
MODULE1_PREDICTOR=distilbert uvicorn service:app --port 8001
```

Nothing downstream changes. That is the point of freezing the contract first.