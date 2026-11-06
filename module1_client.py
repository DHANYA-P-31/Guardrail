"""
Reference client for Module 1 (Intent & Emotion Detection). Module 2 can copy
this file as-is; it only needs `httpx`.

    from module1_client import classify, should_escalate
    resp = classify(user_text)
    escalate, why = should_escalate(resp)

THE RULE THIS FILE EXISTS TO ENFORCE: unknown is never safe. If Module 1 is
down, times out, or returns an error, classify() returns a response with
abstain=True, and should_escalate() treats that as "escalate". A crisis
router that quietly falls back to "normal" when it can't reach the classifier
fails in exactly the situations that matter.

WHAT should_escalate() DOES NOT GUARANTEE: it is necessary, not sufficient.
Measured on the frozen test set, Module 1's crisis_disclosure recall was
2/8 (baseline) and 0/8 (DistilBERT) — the DistilBERT model never predicted
crisis_disclosure once, and on the adversarial set it made confident, non-
abstaining misses on implicit crisis text. Module 2 must run its own checks
on the raw text (keyword/pattern layer from the original spec) and treat
Module 1's output as ONE signal, not the detector.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import httpx

DEFAULT_URL = "http://localhost:8001"
CRISIS_INTENT = "crisis_disclosure"
SUPPORTED_MAJOR = 2   # schema_version "2.x.y"; a different major means the contract changed


def _unavailable(reason: str) -> Dict[str, Any]:
    return {
        "schema_version": None, "model_version": "unavailable", "text_id": None,
        "emotion": None, "intent": None, "calibrated": False,
        "abstain": True, "abstain_reason": f"module1_unavailable: {reason}",
        "latency_ms": None,
    }


def classify(text: str, base_url: str = DEFAULT_URL, timeout: float = 5.0,
             text_id: str | None = None) -> Dict[str, Any]:
    """POST /v1/classify. Never raises: any failure becomes an abstain=True
    response so the caller's normal escalate-on-abstain path handles it."""
    payload: Dict[str, Any] = {"text": text}
    if text_id:
        payload["text_id"] = text_id
    try:
        r = httpx.post(f"{base_url}/v1/classify", json=payload, timeout=timeout)
        r.raise_for_status()
        resp = r.json()
    except (httpx.HTTPError, ValueError) as e:
        return _unavailable(f"{type(e).__name__}: {e}")

    major = str(resp.get("schema_version", "")).split(".")[0]
    if major != str(SUPPORTED_MAJOR):
        return _unavailable(f"schema_version {resp.get('schema_version')!r} not supported (expected {SUPPORTED_MAJOR}.x)")
    return resp


def should_escalate(resp: Dict[str, Any]) -> Tuple[bool, str]:
    """(escalate?, reason). Escalate on abstain OR crisis_disclosure."""
    if resp.get("abstain"):
        return True, resp.get("abstain_reason") or "abstain"
    intent = resp.get("intent") or {}
    if intent.get("label") == CRISIS_INTENT:
        return True, f"intent=crisis_disclosure (confidence {intent.get('confidence'):.2f})"
    return False, "no escalation signal from Module 1 (this does NOT mean the text is safe)"


def to_flat_contract(resp: Dict[str, Any]) -> Dict[str, Any]:
    """The original brief's {emotion, intent, confidence} shape, for a Module 2
    written against that example. ASSUMPTION: 'confidence' = the INTENT head's
    confidence, since intent drives escalation. Labels are also different from
    the brief's example ('depression', 'suicide_risk' don't exist here) — Module
    2's rules must use this module's label set (see GET /health)."""
    emo, intent = resp.get("emotion"), resp.get("intent")
    return {
        "emotion": emo["label"] if emo else None,
        "intent": intent["label"] if intent else None,
        "confidence": intent["confidence"] if intent else 0.0,
        "abstain": bool(resp.get("abstain")),
    }


if __name__ == "__main__":
    import sys

    base = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL   # e.g. python module1_client.py http://localhost:8001
    samples = [
        "What are the common signs of stress?",
        "Thanks, that helped a lot today.",
        "I have decided. Tonight is the night.",
        "asdf qwerty zxcv",
    ]
    for t in samples:
        resp = classify(t, base_url=base)
        esc, why = should_escalate(resp)
        intent = resp["intent"]["label"] if resp.get("intent") else "-"
        print(f"{'ESCALATE' if esc else 'continue':9s} intent={intent:22s} model={resp.get('model_version')} "
              f"calibrated={resp.get('calibrated')}\n          {why}\n          {t!r}")