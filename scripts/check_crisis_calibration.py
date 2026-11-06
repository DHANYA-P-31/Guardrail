"""
Reports confidence statistics specifically on the crisis_disclosure
validation rows, from the logits scripts/export_val_logits.py already
produced — no new model run needed.

Aggregate ECE over the whole validation set is dominated by the two
largest intent classes (emotional_expression, information_seeking — ~95%
of reliable intent rows) and says almost nothing about calibration on a
~4-row minority class. This checks that directly instead of assuming it.

Usage (after scripts/export_val_logits.py and scripts/calibrate.py have run):
    python scripts/check_crisis_calibration.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from service import HIGH_RISK_INTENT, INTENT_LABELS  # noqa: E402

INT_ID = {l: i for i, l in enumerate(INTENT_LABELS)}
CRISIS_ID = INT_ID[HIGH_RISK_INTENT]


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def main():
    data = np.load(ROOT / "artifacts" / "distilbert" / "val_logits.npz")
    intent_logits, intent_labels = data["intent_logits"], data["intent_labels"]

    is_crisis = intent_labels == CRISIS_ID
    n_crisis = int(is_crisis.sum())
    print(f"crisis_disclosure validation rows: {n_crisis}")
    if n_crisis == 0:
        print("None in this split — can't report anything concrete here.")
        return

    crisis_logits = intent_logits[is_crisis]
    probs = softmax(crisis_logits)
    true_class_probs = probs[:, CRISIS_ID]   # model's confidence in the CORRECT answer, on each true crisis row
    predicted = probs.argmax(axis=1)

    print(f"predicted class on each true crisis_disclosure row: "
          f"{[INTENT_LABELS[p] for p in predicted]}")
    print(f"model's confidence in the CORRECT (crisis_disclosure) class per row: "
          f"{[round(float(p), 4) for p in true_class_probs]}")
    print(f"mean confidence in the correct class: {true_class_probs.mean():.4f}")
    print(f"\nCompare this to the aggregate intent ECE from calibrate.py — a low mean "
          f"confidence here, even with low aggregate ECE, confirms the model is "
          f"specifically unreliable on this minority class, which global temperature "
          f"scaling cannot fix (it corrects one scalar across the whole distribution, "
          f"not per-class).")


if __name__ == "__main__":
    main()