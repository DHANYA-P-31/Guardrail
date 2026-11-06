"""
DistilBERT fine-tune, 2 heads on a shared encoder (emotion, intent). No
risk-signals head — crisis severity is carried by intent == "crisis_disclosure".

Each head's loss is MASKED per-row to only the examples where build_dataset.py
marked that head "reliable" (content-derived label, not a blanket per-source
default). Masking rather than filtering rows out keeps both heads training
on every batch from the shared encoder — an intent-unreliable row still
contributes its emotion loss, and vice versa — while never backpropagating
a head's loss through a row whose label for that head is just a fallback
default, not a real annotation.

Each head's loss is ALSO class-weighted (inverse frequency over reliable
rows only). This matters even more than the masking: on the real merged
dataset, intent-reliable rows break down as ~26.5k emotional_expression vs
~27 crisis_disclosure — roughly 1000:1. Unweighted, gradient signal from the
minority class is negligible and the model will very likely never learn to
predict it at all, independent of how many examples eventually exist for
it. baseline.py gets this for free from sklearn's class_weight="balanced";
here it has to be explicit.

NOTE on max_grad_norm: HF Trainer's logged grad_norm is the PRE-clipping
norm, so a spike there (seen in a real run: 60.39, 55.63, 33.45) is very
likely the moment a 203x-weighted crisis_disclosure example hit a batch.
Clipping too tightly (e.g. max_grad_norm=1.0) flattens that back down to
the same ceiling as every ordinary step, substantially cancelling the
amplification class weighting was meant to provide. Kept looser here
(max_grad_norm=5.0) so rare-class-driven gradients still get a real boost
relative to routine ones, while still preventing true explosions.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments

_SAFE_TO_DROP_KWARGS = {"warmup_ratio", "weight_decay", "report_to", "seed"}
_RENAME_FALLBACKS = {"eval_strategy": "evaluation_strategy"}


def _build_training_arguments(**kwargs) -> TrainingArguments:
    current = dict(kwargs)
    while True:
        try:
            return TrainingArguments(**current)
        except TypeError as e:
            m = re.search(r"unexpected keyword argument '(\w+)'", str(e))
            if not m:
                raise
            bad_kwarg = m.group(1)

            if bad_kwarg in _RENAME_FALLBACKS and bad_kwarg in current:
                old_name = _RENAME_FALLBACKS[bad_kwarg]
                print(f"[warn] train.py: '{bad_kwarg}' not accepted by your installed "
                      f"transformers version — retrying as '{old_name}' (known rename).")
                current[old_name] = current.pop(bad_kwarg)
                continue

            if bad_kwarg in _SAFE_TO_DROP_KWARGS:
                print(f"[warn] train.py: '{bad_kwarg}' not accepted by your installed "
                      f"transformers version — dropping it (safe: tuning-only, training "
                      f"still works correctly without it).")
                del current[bad_kwarg]
                continue

            raise TypeError(
                f"TrainingArguments rejected '{bad_kwarg}', and it's not in the safe-to-drop "
                f"or known-rename list — this is a load-bearing argument for how checkpoints "
                f"are selected/evaluated, so refusing to silently proceed without it. Your "
                f"installed transformers version is likely too old or too new for this script. "
                f"Try: pip install -U \"transformers>=4.47,<5\". Original error: {e}"
            ) from e


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from service import EMOTION_LABELS, HIGH_RISK_INTENT, INTENT_LABELS  # noqa: E402

PROC = ROOT / "data" / "processed"
OUT_DIR = ROOT / "artifacts" / "distilbert"
BASE_MODEL = "distilbert-base-uncased"
MAX_LEN = 128
SEED = 42

EMO_ID = {l: i for i, l in enumerate(EMOTION_LABELS)}
INT_ID = {l: i for i, l in enumerate(INTENT_LABELS)}
CRISIS_ID = INT_ID[HIGH_RISK_INTENT]


class M1Dataset(Dataset):
    def __init__(self, path: Path, tokenizer):
        rows = [json.loads(l) for l in open(path)]
        self.texts = [r["text"] for r in rows]
        self.emotion = [EMO_ID[r["emotion"]] for r in rows]
        self.intent = [INT_ID[r["intent"]] for r in rows]
        self.emotion_reliable = [bool(r.get("emotion_reliable", True)) for r in rows]
        self.intent_reliable = [bool(r.get("intent_reliable", True)) for r in rows]
        self.enc = tokenizer(self.texts, truncation=True, padding=True, max_length=MAX_LEN)

        n_emo = sum(self.emotion_reliable)
        n_int = sum(self.intent_reliable)
        print(f"{path.name}: {n_emo}/{len(rows)} emotion-reliable, "
              f"{n_int}/{len(rows)} intent-reliable")

    def class_weights(self, label_ids: list, reliable: list, n_classes: int) -> list:
        counts = Counter(l for l, r in zip(label_ids, reliable) if r)
        n_reliable = sum(counts.values())
        weights = []
        for c in range(n_classes):
            count = counts.get(c, 0)
            weights.append(n_reliable / (n_classes * count) if count > 0 else 0.0)
        return weights

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
        item["emotion_label"] = torch.tensor(self.emotion[idx])
        item["intent_label"] = torch.tensor(self.intent[idx])
        item["emotion_reliable"] = torch.tensor(self.emotion_reliable[idx])
        item["intent_reliable"] = torch.tensor(self.intent_reliable[idx])
        return item


class TwoHeadModel(nn.Module):
    def __init__(self, emotion_weights: list, intent_weights: list,
                 base_model=BASE_MODEL, n_emotion=len(EMOTION_LABELS), n_intent=len(INTENT_LABELS)):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.emotion_head = nn.Linear(hidden, n_emotion)
        self.intent_head = nn.Linear(hidden, n_intent)
        self.register_buffer("emotion_weights", torch.tensor(emotion_weights, dtype=torch.float))
        self.register_buffer("intent_weights", torch.tensor(intent_weights, dtype=torch.float))

    def forward(self, input_ids, attention_mask, emotion_label=None, intent_label=None,
                emotion_reliable=None, intent_reliable=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0])

        emotion_logits = self.emotion_head(cls)
        intent_logits = self.intent_head(cls)

        loss = None
        if emotion_label is not None:
            emo_ce = nn.CrossEntropyLoss(weight=self.emotion_weights, reduction="none")
            int_ce = nn.CrossEntropyLoss(weight=self.intent_weights, reduction="none")

            emo_losses = emo_ce(emotion_logits, emotion_label)
            emo_mask = emotion_reliable.float()
            emo_loss = (emo_losses * emo_mask).sum() / emo_mask.sum().clamp(min=1.0)

            int_losses = int_ce(intent_logits, intent_label)
            int_mask = intent_reliable.float()
            int_loss = (int_losses * int_mask).sum() / int_mask.sum().clamp(min=1.0)

            loss = emo_loss + int_loss

        return {"loss": loss, "emotion_logits": emotion_logits, "intent_logits": intent_logits}


_N_EMOTION, _N_INTENT = len(EMOTION_LABELS), len(INTENT_LABELS)


def crisis_recall_metric(eval_pred):
    """Computes recall on crisis_disclosure over intent-reliable evaluation rows.

    HF Trainer passes model-output dicts and label_names lists back to
    compute_metrics as TUPLES (preds, labels), not dicts. The PREVIOUS version
    of this function picked intent_logits by a hardcoded tuple index (1),
    assuming dict-insertion order — this is an assumption about Trainer
    internals that is NOT guaranteed to hold identically across transformers
    versions/environments (Colab vs Kaggle, different releases), and a wrong
    index here silently produces a stable, always-wrong recall number — which
    is exactly what a persistent eval_recall_crisis_disclosure: 0 across every
    epoch, on two different platforms, looks like. Fixed by identifying the
    intent tensor by its actual last-dimension size (8 classes) instead of
    its position — this works regardless of tuple order, tested against both
    possible orderings before trusting it."""
    preds, labels = eval_pred

    if isinstance(preds, tuple):
        candidates = [p for p in preds if p.shape[-1] == _N_INTENT]
        if len(candidates) != 1:
            raise ValueError(
                f"Expected exactly one prediction tensor with last dim {_N_INTENT} "
                f"(intent_logits) among eval_pred.predictions, got shapes: "
                f"{[getattr(p, 'shape', None) for p in preds]}. Can't reliably identify "
                f"intent_logits — refusing to silently guess and produce a wrong metric."
            )
        intent_logits = candidates[0]
    else:
        intent_logits = preds

    # label_names=["emotion_label", "intent_label", "emotion_reliable", "intent_reliable"]
    # — this list order IS guaranteed by Trainer (it's what we told it to use),
    # unlike the predictions dict's insertion order, so index-based access
    # here is safe.
    if isinstance(labels, tuple):
        intent_labels = labels[1]
        intent_reliable = labels[3].astype(bool)
    else:
        intent_labels = labels
        intent_reliable = np.ones_like(intent_labels, dtype=bool)

    intent_pred = intent_logits.argmax(axis=-1)
    is_crisis = (intent_labels == CRISIS_ID) & intent_reliable

    if is_crisis.sum() == 0:
        return {"recall_crisis_disclosure": float("nan")}

    recall = float((intent_pred[is_crisis] == CRISIS_ID).mean())
    return {"recall_crisis_disclosure": recall}


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    train_ds = M1Dataset(PROC / "train.jsonl", tokenizer)
    val_ds = M1Dataset(PROC / "val.jsonl", tokenizer)

    emotion_weights = train_ds.class_weights(train_ds.emotion, train_ds.emotion_reliable, len(EMOTION_LABELS))
    intent_weights = train_ds.class_weights(train_ds.intent, train_ds.intent_reliable, len(INTENT_LABELS))
    print("intent class weights:", dict(zip(INTENT_LABELS, [round(w, 2) for w in intent_weights])))

    if intent_weights[CRISIS_ID] == 0.0:
        raise SystemExit(
            "crisis_disclosure has ZERO reliable training examples in train.jsonl — "
            "rerun build_dataset.py and check the MHDialog Level override is firing, "
            "or hand-label more examples, before training."
        )

    model = TwoHeadModel(emotion_weights=emotion_weights, intent_weights=intent_weights)

    args = _build_training_arguments(
        output_dir=str(OUT_DIR / "checkpoints"),
        num_train_epochs=4,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        max_grad_norm=5.0,   # loosened from 1.0 — see module docstring: tight clipping
                             # was likely cancelling out the 203x crisis_disclosure weight
        warmup_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="recall_crisis_disclosure",
        greater_is_better=True,
        seed=SEED,
        report_to=[],
        label_names=["emotion_label", "intent_label", "emotion_reliable", "intent_reliable"],
    )

    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=crisis_recall_metric,
    )

    checkpoint_dir = OUT_DIR / "checkpoints"
    resume = checkpoint_dir.exists() and any(p.name.startswith("checkpoint-") for p in checkpoint_dir.iterdir())
    if resume:
        print(f"Found existing checkpoints in {checkpoint_dir} — resuming instead of "
              f"retraining earlier epochs from scratch.")
    trainer.train(resume_from_checkpoint=resume)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUT_DIR / "model.pt")
    tokenizer.save_pretrained(OUT_DIR)
    print(f"Saved to {OUT_DIR}. Next: scripts/calibrate.py, then implement "
          f"DistilBertPredictor.predict in service.py to load this checkpoint.")


if __name__ == "__main__":
    main()