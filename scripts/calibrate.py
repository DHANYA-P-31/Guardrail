"""
Temperature scaling: fits one scalar T per head so softmax(logits/T) is
calibrated on held-out data. Fine-tuned transformers are reliably
overconfident, so an uncalibrated max-probability is not a real likelihood —
Module 2 cannot threshold on it safely until this is fitted.

Usage (after scripts/train.py has produced val logits):
    python scripts/calibrate.py

This file is runnable standalone right now (see __main__) against synthetic
logits, to prove the math is correct before real logits exist.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def softmax(x: np.ndarray, axis=-1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Mean gap between confidence and accuracy across confidence bins."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(float)

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = correct[mask].mean()
        ece += (mask.sum() / len(labels)) * abs(bin_conf - bin_acc)
    return float(ece)


_MAX_TEMPERATURE = 20.0   # see note below — the local refine has no inherent
                          # upper bound and can run away on degenerate data


def fit_temperature(logits: np.ndarray, labels: np.ndarray, lr: float = 0.01, steps: int = 200) -> float:
    """Grid + local refine — no torch dependency needed for one scalar.

    Tested against synthetic all-random logits (zero real signal between
    logits and labels): the local refine ran past the grid's own upper
    bound (5.0) all the way to 15.0, because driving T toward infinity
    genuinely minimises NLL when there's nothing to calibrate against —
    softmax(logits/T) flattens toward uniform, which IS the right answer
    for pure noise. On real model logits this shouldn't happen (there's
    actual signal to calibrate), so a fitted T that still hits the cap is
    informative: either the logits are near-degenerate for that class, or
    something upstream is wrong — surfaced as a warning rather than
    silently accepted."""
    def nll(T: float) -> float:
        probs = softmax(logits / T)
        p_true = probs[np.arange(len(labels)), labels]
        return -np.log(np.clip(p_true, 1e-12, 1.0)).mean()

    grid_min, grid_max = 0.5, 5.0
    candidates = np.linspace(grid_min, grid_max, 91)
    losses = [nll(T) for T in candidates]
    best_T = float(candidates[int(np.argmin(losses))])

    # local refine around the grid optimum, capped so it can't run away
    step = 0.05
    for _ in range(steps):
        improved = False
        for cand in (best_T - step, best_T + step):
            if 0 < cand <= _MAX_TEMPERATURE and nll(cand) < nll(best_T):
                best_T = cand
                improved = True
        if not improved:
            step *= 0.5
        if step < 1e-4:
            break

    if best_T > grid_max:
        # The refine step pushed past where the coarse grid search already
        # covered well — tested and confirmed this happens on purely random
        # (zero-signal) logits, where T -> large genuinely minimises NLL.
        # On real model logits this is a red flag, not expected behaviour.
        print(f"[warn] calibrate.py: fitted temperature ({best_T:.2f}) is past the grid "
              f"search's own range ({grid_min}-{grid_max}) — this usually means the "
              f"logits for this head carry little real signal (near-random), not a "
              f"healthy calibration fit. Worth checking this head's logits/labels "
              f"before trusting the result.")
    return best_T


def calibrate_head(logits: np.ndarray, labels: np.ndarray, name: str) -> dict:
    raw_probs = softmax(logits)
    raw_ece = expected_calibration_error(raw_probs, labels)

    T = fit_temperature(logits, labels)
    scaled_probs = softmax(logits / T)
    scaled_ece = expected_calibration_error(scaled_probs, labels)

    return {
        "head": name, "temperature": round(T, 4),
        "ece_before": round(raw_ece, 4), "ece_after": round(scaled_ece, 4),
    }


def _make_synthetic_overconfident_logits(n=2000, n_classes=7, seed=42):
    """Simulates the failure mode we're correcting: a model that's right
    ~70% of the time but reports >95% confidence almost always."""
    rng = np.random.RandomState(seed)
    labels = rng.randint(0, n_classes, size=n)
    logits = rng.normal(0, 1, size=(n, n_classes))
    correct_mask = rng.rand(n) < 0.70
    for i in range(n):
        target = labels[i] if correct_mask[i] else rng.randint(0, n_classes)
        logits[i, target] += 6.0   # inflate one logit -> overconfident peak
    return logits, labels


if __name__ == "__main__":
    art = Path(__file__).resolve().parent.parent / "artifacts"
    logits_path = art / "distilbert" / "val_logits.npz"

    if logits_path.exists():
        data = np.load(logits_path)
        results = [
            calibrate_head(data["emotion_logits"], data["emotion_labels"], "emotion"),
            calibrate_head(data["intent_logits"], data["intent_labels"], "intent"),
        ]
        (art / "distilbert" / "calibration.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(results, indent=2))
    else:
        print("No trained-model logits found yet — demonstrating on synthetic "
              "overconfident logits to prove the calibration math is correct.\n")
        logits, labels = _make_synthetic_overconfident_logits()
        result = calibrate_head(logits, labels, "synthetic_demo")
        print(json.dumps(result, indent=2))
        assert result["ece_after"] < result["ece_before"], "calibration should reduce ECE"
        print("\nECE reduced by calibration, as expected. Re-run this script for "
              "real once scripts/train.py has produced val_logits.npz.")