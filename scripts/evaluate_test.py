"""
Final evaluation on the FROZEN test set (data/processed/test.jsonl), for the
TF-IDF baseline and the DistilBERT checkpoint, using the project's own
evaluation policy: per-class metrics and macro-F1 rather than accuracy, and
crisis_disclosure recall reported WITH its sample size and a confidence
interval — never as a bare number.

Usage:
    python scripts/evaluate_test.py

This is meant to be run once, at the end. Every run is appended to
results/test_eval_log.jsonl and the run number is printed, so the report can
say honestly how many times the test set was looked at. Don't tune anything
(thresholds, class weights, checkpoints) based on what this prints — that
turns the test set into a second validation set.

Metrics are top-1 (argmax), so they don't depend on temperature scaling or
ABSTAIN_THRESHOLD: dividing logits by a positive scalar never changes argmax.

Each head is evaluated only on rows where that head's label is reliable
(emotion_reliable / intent_reliable), same rule as training.

DistilBERT is skipped cleanly if its artifacts or torch aren't available.
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from service import EMOTION_LABELS, HIGH_RISK_INTENT, INTENT_LABELS  # noqa: E402

PROC = ROOT / "data" / "processed"
RES = ROOT / "results"


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return ((centre - margin) / denom, (centre + margin) / denom)


def head_report(y_true, y_pred, labels):
    """Per-class precision/recall/F1/support plus macro-F1 over classes that
    actually have test support (a class with zero support has no defined
    recall, and including it as 0 would unfairly drag the macro average)."""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    ids = list(range(len(labels)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=ids, zero_division=0)
    per_class = {
        labels[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
        for i in ids
    }
    supported = [i for i in ids if s[i] > 0]
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1_over_supported_classes": float(np.mean([f[i] for i in supported])) if supported else float("nan"),
        "n_classes_with_support": len(supported),
        "per_class": per_class,
    }


def crisis_report(y_true, y_pred):
    cid = INTENT_LABELS.index(HIGH_RISK_INTENT)
    mask = np.array(y_true) == cid
    n = int(mask.sum())
    k = int((np.array(y_pred)[mask] == cid).sum())
    lo, hi = wilson(k, n)
    predicted_as = Counter(INTENT_LABELS[p] for p in np.array(y_pred)[mask])
    n_pred_crisis = int((np.array(y_pred) == cid).sum())
    n_pred_crisis_correct = int(((np.array(y_pred) == cid) & mask).sum())
    return {
        "support": n, "recall_hits": k,
        "recall": (k / n) if n else float("nan"),
        "recall_95ci_wilson": [lo, hi],
        "true_crisis_rows_predicted_as": dict(predicted_as),
        "times_predicted_crisis": n_pred_crisis,
        "times_predicted_crisis_correctly": n_pred_crisis_correct,
    }


def eval_baseline(texts, emo_y, int_y, emo_mask, int_mask):
    import joblib
    d = ROOT / "artifacts" / "baseline"
    need = ["vectorizer.joblib", "emotion_clf.joblib", "intent_clf.joblib"]
    missing = [f for f in need if not (d / f).exists()]
    if missing:
        raise FileNotFoundError(f"missing baseline artifacts {missing} — run scripts/baseline.py first")
    vec = joblib.load(d / "vectorizer.joblib")
    emo_clf = joblib.load(d / "emotion_clf.joblib")
    int_clf = joblib.load(d / "intent_clf.joblib")
    X = vec.transform(texts)
    emo_pred_names = emo_clf.predict(X)
    int_pred_names = int_clf.predict(X)
    emo_pred = np.array([EMOTION_LABELS.index(n) for n in emo_pred_names])
    int_pred = np.array([INTENT_LABELS.index(n) for n in int_pred_names])
    return "m1-v0.2-baseline-tfidf-lr", emo_pred, int_pred


def eval_distilbert(texts, batch_size: int = 64):
    import torch
    from service import DistilBertPredictor
    pred = DistilBertPredictor()
    print(f"  loaded {pred.model_version} on {pred.device}")
    order = np.argsort([len(t) for t in texts])   # similar lengths batch together -> far less padding
    emo_pred = np.zeros(len(texts), dtype=np.int64)
    int_pred = np.zeros(len(texts), dtype=np.int64)
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            idx = order[start:start + batch_size]
            enc = pred.tokenizer([texts[i] for i in idx], truncation=True, max_length=pred._max_len,
                                 padding=True, return_tensors="pt")
            e_logits, i_logits = pred._model(input_ids=enc["input_ids"].to(pred.device),
                                             attention_mask=enc["attention_mask"].to(pred.device))
            emo_pred[idx] = e_logits.argmax(-1).cpu().numpy()
            int_pred[idx] = i_logits.argmax(-1).cpu().numpy()
            if (start // batch_size) % 50 == 0:
                print(f"  {start}/{len(texts)}")
    return pred.model_version, emo_pred, int_pred


def main():
    test_path = PROC / "test.jsonl"
    manifest_path = PROC / "split_manifest.json"
    if not test_path.exists():
        raise SystemExit(f"{test_path} not found — run scripts/build_dataset.py first.")
    rows = [json.loads(l) for l in open(test_path)]
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if manifest.get("n_test") not in (None, len(rows)):
        raise SystemExit(f"test.jsonl has {len(rows)} rows but split_manifest says {manifest['n_test']} — "
                         f"the split has changed since it was frozen; results wouldn't be comparable.")

    RES.mkdir(exist_ok=True)
    log = RES / "test_eval_log.jsonl"
    run_no = (sum(1 for _ in open(log)) if log.exists() else 0) + 1
    print(f"Test set: {len(rows)} rows, sha256 {str(manifest.get('test_set_sha256'))[:16]}…")
    print(f"This is test-set evaluation run #{run_no}.\n")

    texts = [r["text"] for r in rows]
    emo_true = np.array([EMOTION_LABELS.index(r["emotion"]) for r in rows])
    int_true = np.array([INTENT_LABELS.index(r["intent"]) for r in rows])
    emo_mask = np.array([bool(r.get("emotion_reliable", True)) for r in rows])
    int_mask = np.array([bool(r.get("intent_reliable", True)) for r in rows])
    print(f"emotion-reliable rows: {int(emo_mask.sum())}   intent-reliable rows: {int(int_mask.sum())}\n")

    results = {}
    for name, fn in (("baseline", lambda: eval_baseline(texts, emo_true, int_true, emo_mask, int_mask)),
                     ("distilbert", lambda: eval_distilbert(texts))):
        print(f"=== {name} ===")
        try:
            version, emo_pred, int_pred = fn()
        except (FileNotFoundError, ImportError) as e:
            print(f"  skipped: {e}\n")
            continue
        results[name] = {
            "model_version": version,
            "emotion": head_report(emo_true[emo_mask], emo_pred[emo_mask], EMOTION_LABELS),
            "intent": head_report(int_true[int_mask], int_pred[int_mask], INTENT_LABELS),
            "crisis_disclosure": crisis_report(int_true[int_mask], int_pred[int_mask]),
        }
        if name == "distilbert":
            results[name]["training_epoch"] = "unknown (see MODEL_CARD.md)"
        r = results[name]
        print(f"  {version}")
        print(f"  emotion  n={r['emotion']['n']}  accuracy={r['emotion']['accuracy']:.3f}  "
              f"macro-F1={r['emotion']['macro_f1_over_supported_classes']:.3f}")
        print(f"  intent   n={r['intent']['n']}  accuracy={r['intent']['accuracy']:.3f}  "
              f"macro-F1={r['intent']['macro_f1_over_supported_classes']:.3f} "
              f"(over {r['intent']['n_classes_with_support']} classes with support)")
        c = r["crisis_disclosure"]
        lo, hi = c["recall_95ci_wilson"]
        print(f"  crisis_disclosure: recall {c['recall_hits']}/{c['support']}"
              + (f" = {c['recall']:.2f}  (95% CI {lo:.2f}–{hi:.2f})" if c["support"] else " — no test rows"))
        print(f"    true crisis rows were predicted as: {c['true_crisis_rows_predicted_as']}")
        print(f"    model predicted crisis {c['times_predicted_crisis']}x, "
              f"{c['times_predicted_crisis_correctly']} correct")
        print("  per-class intent (precision / recall / F1 / support):")
        for lab, m in r["intent"]["per_class"].items():
            print(f"    {lab:22s} {m['precision']:.2f} / {m['recall']:.2f} / {m['f1']:.2f} / {m['support']}")
        print()

    if not results:
        raise SystemExit("No model could be evaluated.")

    out = {"run": run_no, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
           "test_set_sha256": manifest.get("test_set_sha256"), "n_test": len(rows), "results": results}
    (RES / "test_eval.json").write_text(json.dumps(out, indent=2))
    with open(log, "a") as fh:
        fh.write(json.dumps({"run": run_no, "timestamp": out["timestamp"], "models": list(results)}) + "\n")
    print(f"Saved results/test_eval.json (run #{run_no}).")
    if "distilbert" in results:
        print("\nReminder for the report: DistilBERT's epoch is unknown (probably epoch 1) — "
              "say so next to every DistilBERT number.")


if __name__ == "__main__":
    main()