"""
Runs the trained DistilBERT checkpoint over the real validation set and saves
raw logits + integer labels to artifacts/distilbert/val_logits.npz, so
scripts/calibrate.py can fit temperature scaling against real numbers
instead of its synthetic demo.

Usage (after you have a trained checkpoint at artifacts/distilbert/model.pt):
    python scripts/export_val_logits.py

Only RELIABLE rows are exported per head — see build_dataset.py's
emotion_reliable/intent_reliable flags. Calibrating against a head's
fallback-default label (e.g. every dair-ai row's unreliable "emotional_expression"
intent) would fit the temperature to noise, not to the model's actual
miscalibration on real annotations. The two heads can end up with different
row counts for exactly this reason — that's correct, not a bug.

Saves RAW logits, not softmax probabilities — temperature scaling divides
logits by T before the softmax, so calibrate.py needs the pre-softmax
values.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PROC = ROOT / "data" / "processed"
OUT = ROOT / "artifacts" / "distilbert" / "val_logits.npz"


def main():
    import torch
    from service import DistilBertPredictor, EMOTION_LABELS, INTENT_LABELS

    EMO_ID = {l: i for i, l in enumerate(EMOTION_LABELS)}
    INT_ID = {l: i for i, l in enumerate(INTENT_LABELS)}

    predictor = DistilBertPredictor()  # raises clearly if artifacts/distilbert/model.pt is missing
    print(f"Loaded {predictor.model_version}, device={predictor.device}")

    rows = [json.loads(l) for l in open(PROC / "val.jsonl")]
    print(f"Running inference on {len(rows)} validation rows...")

    emo_logits_list, emo_labels_list = [], []
    int_logits_list, int_labels_list = [], []

    with torch.no_grad():
        for i, row in enumerate(rows):
            if i % 2000 == 0:
                print(f"  {i}/{len(rows)}")

            enc = predictor.tokenizer(row["text"], truncation=True, max_length=predictor._max_len, return_tensors="pt")
            input_ids = enc["input_ids"].to(predictor.device)
            attention_mask = enc["attention_mask"].to(predictor.device)
            emotion_logits, intent_logits = predictor._model(input_ids=input_ids, attention_mask=attention_mask)

            if row.get("emotion_reliable", True):
                emo_logits_list.append(emotion_logits[0].cpu().numpy())
                emo_labels_list.append(EMO_ID[row["emotion"]])
            if row.get("intent_reliable", True):
                int_logits_list.append(intent_logits[0].cpu().numpy())
                int_labels_list.append(INT_ID[row["intent"]])

    emotion_logits_arr = np.stack(emo_logits_list)
    emotion_labels_arr = np.array(emo_labels_list, dtype=np.int64)
    intent_logits_arr = np.stack(int_logits_list)
    intent_labels_arr = np.array(int_labels_list, dtype=np.int64)

    print(f"\nemotion: {len(emotion_labels_arr)} reliable rows exported")
    print(f"intent:  {len(intent_labels_arr)} reliable rows exported")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT,
        emotion_logits=emotion_logits_arr, emotion_labels=emotion_labels_arr,
        intent_logits=intent_logits_arr, intent_labels=intent_labels_arr,
    )
    print(f"Saved to {OUT}")
    print("Next: python scripts/calibrate.py")


if __name__ == "__main__":
    main()