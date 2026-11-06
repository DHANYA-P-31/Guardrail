"""
TF-IDF + logistic regression baseline. Exists so the DistilBERT fine-tune has
something concrete to beat — without this you cannot claim the transformer
was necessary.

Two heads only (emotion, intent) — matches the spec exactly, no risk-signal
head. Headline metric is recall on intent == "crisis_disclosure", not
accuracy: a missed crisis disclosure is categorically worse than a false
positive.

Each head trains ONLY on rows where build_dataset.py marked that head
"reliable" — i.e. the label came from real per-example content (a topic tag,
a pattern-matched question, a risk-severity override), not a blanket
per-source default. Training on the full merged set regardless would teach
the intent head "always guess emotional_expression" from the ~99% of rows
where that's just the fallback label, not an annotation.

Usage:
    python scripts/baseline.py

Reads data/processed/{train,val,test}.jsonl (run build_dataset.py first).
Writes results/baseline_metrics.json and artifacts/baseline/*.joblib.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from service import HIGH_RISK_INTENT  # noqa: E402

PROC = ROOT / "data" / "processed"
ART = ROOT / "artifacts" / "baseline"
RES = ROOT / "results"
ART.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)


def load_split(name: str) -> pd.DataFrame:
    path = PROC / f"{name}.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} not found — run scripts/build_dataset.py first.")
    return pd.read_json(path, lines=True)


def train_head(X_train_text, y_train, X_val_text, y_val, vectorizer):
    Xtr = vectorizer.transform(X_train_text)
    Xval = vectorizer.transform(X_val_text)
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)
    clf.fit(Xtr, y_train)
    preds = clf.predict(Xval)
    return clf, preds


def main():
    train = load_split("train")
    val = load_split("val")

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, max_features=50000)
    vectorizer.fit(train["text"])

    metrics = {"note": "Baseline metrics. Compare against DistilBERT on the same split_manifest test set."}

    train_emo = train[train.get("emotion_reliable", True)]
    val_emo = val[val.get("emotion_reliable", True)]
    train_int = train[train.get("intent_reliable", True)]
    val_int = val[val.get("intent_reliable", True)]

    print(f"emotion head: training on {len(train_emo)}/{len(train)} rows "
          f"({len(train_emo) / max(len(train), 1) * 100:.1f}% have reliable emotion labels)")
    print(f"intent head:  training on {len(train_int)}/{len(train)} rows "
          f"({len(train_int) / max(len(train), 1) * 100:.1f}% have reliable intent labels)")

    if val_emo.empty or val_int.empty:
        raise SystemExit(
            "Validation split has zero reliable rows for one of the heads — "
            "cannot compute metrics. Check build_dataset.py's reliability flags "
            "and your source mix."
        )

    emo_clf, emo_preds = train_head(train_emo["text"], train_emo["emotion"], val_emo["text"], val_emo["emotion"], vectorizer)
    metrics["emotion"] = classification_report(val_emo["emotion"], emo_preds, output_dict=True, zero_division=0)
    metrics["emotion"]["_n_train_rows"] = len(train_emo)
    joblib.dump(emo_clf, ART / "emotion_clf.joblib")

    int_clf, int_preds = train_head(train_int["text"], train_int["intent"], val_int["text"], val_int["intent"], vectorizer)
    metrics["intent"] = classification_report(val_int["intent"], int_preds, output_dict=True, zero_division=0)
    metrics["intent"]["_n_train_rows"] = len(train_int)
    joblib.dump(int_clf, ART / "intent_clf.joblib")
    joblib.dump(vectorizer, ART / "vectorizer.joblib")

    # Headline: recall on crisis_disclosure — NOT overall accuracy.
    crisis_metrics = metrics["intent"].get(HIGH_RISK_INTENT, {})
    metrics["headline_crisis_recall"] = {
        f"{HIGH_RISK_INTENT}_recall": crisis_metrics.get("recall"),
        f"{HIGH_RISK_INTENT}_precision": crisis_metrics.get("precision"),
        f"{HIGH_RISK_INTENT}_support": crisis_metrics.get("support"),
    }

    (RES / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    print(f"\n=== Headline: recall on {HIGH_RISK_INTENT} (report this, not accuracy) ===")
    for k, v in metrics["headline_crisis_recall"].items():
        print(f"  {k}: {v}")
    print(f"\nFull metrics written to {RES / 'baseline_metrics.json'}")
    print(f"Models saved to {ART}/")


if __name__ == "__main__":
    main()