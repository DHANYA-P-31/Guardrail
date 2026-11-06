"""
Module 1 — Intent & Emotion Detection. Single-file service.

Matches the spec exactly: two classification heads (emotion, intent), each
with its own label and confidence. No separate risk-signal head — crisis
severity is carried entirely by intent == "crisis_disclosure", same as the
brief's own worked example.

Run:  uvicorn service:app --reload --port 8001
Docs: http://localhost:8001/docs

This is a router, not a clinical instrument. When confidence is too low to
trust, the service abstains and Module 2 must escalate, never default to
"normal".
"""

from __future__ import annotations

import os
import re
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

# Artifact lookup is anchored to THIS file, not the working directory — the
# service gets started by other people / process managers from arbitrary
# folders. service.py may live at the project root OR inside scripts/ (it does
# on the author's machine), so look next to it, then one level up, then in the
# working directory. MODULE1_ARTIFACTS_DIR overrides all of that, strictly.
_HERE = Path(__file__).resolve().parent


def _artifact_candidates(name: str):
    env = os.getenv("MODULE1_ARTIFACTS_DIR")
    if env:
        return [Path(env) / name]
    return [_HERE / "artifacts" / name, _HERE.parent / "artifacts" / name, Path.cwd() / "artifacts" / name]


def _artifact_dir(name: str) -> Path:
    cands = _artifact_candidates(name)
    return next((c for c in cands if c.is_dir()), cands[0])

# ─────────────────────────────────────────────────────────────────────────
# LABEL SCHEMA — single source of truth, frozen against the spec you pasted.
# Don't change strings here without bumping SCHEMA_VERSION and telling
# Member 2 (Crisis Risk Engine) — he keys off exactly these strings.
# ─────────────────────────────────────────────────────────────────────────

SCHEMA_VERSION = "2.0.0"   # bumped: risk_signals head removed, taxonomy expanded

EMOTION_LABELS = [
    "joy", "sadness", "anger", "fear", "love_care", "surprise",
    "confusion", "guilt_remorse", "hope_optimism", "neutral",
]

INTENT_LABELS = [
    "emotional_expression",
    "emotional_support",     # asking for comfort/presence
    "advice_seeking",
    "information_seeking",
    "help_seeking",
    "problem_description",
    "crisis_disclosure",     # imminent self-harm / suicidal intent — outranks all others
    "casual_conversation",
]

# intent == crisis_disclosure is the one class where a false negative is
# unacceptable. Recall on this class, not accuracy, is the headline metric.
HIGH_RISK_INTENT = "crisis_disclosure"

# Below this calibrated max-probability, abstain and let Module 2 escalate.
# Placeholder — retune empirically once calibration (week 3) is done.
ABSTAIN_THRESHOLD = 0.45


# ─────────────────────────────────────────────────────────────────────────
# CONTRACT — frozen I/O shared with Module 2. Matches the brief's worked
# example: {"emotion": ..., "intent": ..., "confidence": ...} — extended
# with the full distribution per head so Module 2 can set its own
# thresholds instead of trusting our argmax blindly.
# ─────────────────────────────────────────────────────────────────────────

class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    text_id: Optional[str] = None
    locale: str = "en"

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank")
        return v


class Prediction(BaseModel):
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    distribution: Dict[str, float]   # full dist, so Module 2 sets its own thresholds


class ClassifyResponse(BaseModel):
    schema_version: str = SCHEMA_VERSION
    model_version: str
    text_id: str

    emotion: Prediction
    intent: Prediction

    calibrated: bool   # False -> treat probabilities as ordinal only
    abstain: bool      # True -> Module 2 MUST escalate, never default to normal
    abstain_reason: Optional[str] = None

    latency_ms: float


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_version: str
    schema_version: str
    calibrated: bool
    labels: Dict[str, List[str]]


# ─────────────────────────────────────────────────────────────────────────
# PREDICTOR — interface + stub. The stub is a transparent keyword heuristic,
# not a model. It must never appear in an evaluation table.
# ─────────────────────────────────────────────────────────────────────────

class Predictor(ABC):
    model_version: str = "unset"
    calibrated: bool = False

    @abstractmethod
    def predict(self, text: str) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Return (emotion_dist, intent_dist) — both softmax, sum to 1."""

    def should_abstain(self, emotion: Dict[str, float], intent: Dict[str, float]) -> Tuple[bool, Optional[str]]:
        top_intent = max(intent.values())
        top_emotion = max(emotion.values())
        if top_intent < ABSTAIN_THRESHOLD:
            return True, f"intent confidence {top_intent:.2f} below {ABSTAIN_THRESHOLD}"
        if top_emotion < ABSTAIN_THRESHOLD:
            return True, f"emotion confidence {top_emotion:.2f} below {ABSTAIN_THRESHOLD}"
        return False, None


def _uniform(labels: List[str]) -> Dict[str, float]:
    return {label: round(1.0 / len(labels), 4) for label in labels}


def _peak(labels: List[str], winner: str, mass: float) -> Dict[str, float]:
    rest = (1.0 - mass) / (len(labels) - 1)
    return {label: round(mass if label == winner else rest, 4) for label in labels}


class StubPredictor(Predictor):
    model_version = "m1-v0.2-stub-DO-NOT-EVALUATE"
    calibrated = False

    _CRISIS_PATTERN = r"\b(kill myself|end my life|don'?t want to live|suicid\w*|hurt myself|self[- ]harm)\b"
    _ANXIETY_PATTERN = r"\b(anxious|anxiety|panic|worried|can'?t sleep)\b"
    _INFO_PATTERN = r"\b(what|how|why|signs of|symptoms)\b"
    _CASUAL_PATTERN = r"\b(hi|hello|hey|thanks|thank you)\b"

    def predict(self, text: str):
        low = text.lower()

        if re.search(self._CRISIS_PATTERN, low):
            emotion = _peak(EMOTION_LABELS, "sadness", 0.60)
            intent = _peak(INTENT_LABELS, "crisis_disclosure", 0.66)
        elif re.search(self._ANXIETY_PATTERN, low):
            emotion = _peak(EMOTION_LABELS, "fear", 0.55)
            intent = _peak(INTENT_LABELS, "emotional_support", 0.50)
        elif re.search(self._INFO_PATTERN, low):
            emotion = _peak(EMOTION_LABELS, "neutral", 0.55)
            intent = _peak(INTENT_LABELS, "information_seeking", 0.58)
        elif re.search(self._CASUAL_PATTERN, low):
            emotion = _peak(EMOTION_LABELS, "neutral", 0.50)
            intent = _peak(INTENT_LABELS, "casual_conversation", 0.55)
        else:
            # Unknown input -> uniform -> abstain. Correct default for a
            # safety router: unsure means escalate, not "normal".
            emotion = _uniform(EMOTION_LABELS)
            intent = _uniform(INTENT_LABELS)

        return emotion, intent


class BaselinePredictor(Predictor):
    """TF-IDF + logistic regression baseline, loaded from the artifacts
    scripts/baseline.py produces. Real trained model, not a heuristic —
    unlike the stub, this one's outputs are meaningful to evaluate."""

    model_version = "m1-v0.2-baseline-tfidf-lr"
    calibrated = False   # sklearn's predict_proba is not temperature-scaled; don't conflate with the fitted-calibration meaning used elsewhere

    def __init__(self, artifacts_dir: str | None = None):
        import joblib   # lazy import — stub-only usage shouldn't require scikit-learn/joblib installed

        d = Path(artifacts_dir) if artifacts_dir else _artifact_dir("baseline")
        missing = [f for f in ("vectorizer.joblib", "emotion_clf.joblib", "intent_clf.joblib") if not (d / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing baseline artifact(s) in {d}: {missing}. "
                f"Run scripts/baseline.py first. Searched: {[str(c) for c in _artifact_candidates('baseline')]}"
            )
        self.vectorizer = joblib.load(d / "vectorizer.joblib")
        self.emotion_clf = joblib.load(d / "emotion_clf.joblib")
        self.intent_clf = joblib.load(d / "intent_clf.joblib")

    def _dist(self, clf, X, label_space: List[str]) -> Dict[str, float]:
        """sklearn's predict_proba is ordered by clf.classes_, not our
        canonical label order, and a class absent from training data (e.g.
        if a future rerun has zero examples for one label) simply won't
        appear in classes_ at all — reindex onto the full label space,
        filling missing entries with 0.0, then renormalise defensively."""
        proba = clf.predict_proba(X)[0]
        raw = dict(zip(clf.classes_, proba))
        full = {label: raw.get(label, 0.0) for label in label_space}
        total = sum(full.values())
        if total <= 0:
            return _uniform(label_space)   # degenerate case: classifier has no signal for any label in this space
        return {label: v / total for label, v in full.items()}

    def predict(self, text: str):
        X = self.vectorizer.transform([text])
        emotion_dist = self._dist(self.emotion_clf, X, EMOTION_LABELS)
        intent_dist = self._dist(self.intent_clf, X, INTENT_LABELS)
        return emotion_dist, intent_dist


class DistilBertPredictor(Predictor):
    """Fine-tuned DistilBERT, shared encoder + two independent softmax heads,
    loaded from the checkpoint scripts/train.py produces.

    HISTORY WORTH KEEPING: an earlier Colab run hit a resume-from-checkpoint
    bug that produced a corrupted checkpoint (global_step already at
    max_steps, 0 seconds of actual training) after a manual interrupt at
    ~89% through epoch 4 — that run's weights came from the last verified-
    legitimate checkpoint, end of epoch 3, and model_version was
    "m1-v1.0-distilbert-3epoch" to report that honestly. ALSO found and
    fixed during that debugging: max_grad_norm=1.0 was likely cancelling
    out the 203x crisis_disclosure loss weight (loosened to 5.0), and
    crisis_recall_metric picked intent_logits by a hardcoded tuple index
    that wasn't guaranteed stable across transformers versions (fixed to
    identify it by last-dim size instead — see scripts/train.py).

    THIS is a different, later run: Kaggle, clean restart (not resumed), all
    4 epochs completed with both fixes in place. BUT which epoch's weights
    model.pt actually holds is NOT verified, and is probably epoch 1: that
    run selected its "best" checkpoint on recall_crisis_disclosure, which
    was exactly 0 at every epoch (only 4 validation rows), and Trainer only
    replaces its best on a strictly greater value, so ties would have left
    epoch 1 as best and load_best_model_at_end would have restored it before
    model.pt was saved. (An earlier version of this docstring also claimed
    eval loss "dropped monotonically" — wrong: it was 1.755, 1.675, 1.606,
    then 1.74 at epoch 4.) Run scripts/identify_checkpoint.py to find out
    which checkpoint model.pt matches. The Kaggle checkpoints were NOT kept, so
    that cannot be settled without retraining. model_version makes no epoch claim;
    the fact that the epoch is unknown travels in `training_epoch`, which the
    evaluation reports include. Do not call this "4 epochs" in a report."""

    model_version = "m1-v1.0-distilbert-kaggle"
    # Kept OUT of model_version on purpose, but not dropped: reports carry it as its own field.
    training_epoch = "unknown"   # Kaggle checkpoints were not kept; probably epoch 1, see MODEL_CARD.md

    def __init__(self, checkpoint_dir: str | None = None):
        import torch
        import torch.nn as nn
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        d = Path(checkpoint_dir) if checkpoint_dir else _artifact_dir("distilbert")
        model_path = d / "model.pt"
        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model at {model_path} (searched: {[str(c) for c in _artifact_candidates('distilbert')]}). Run scripts/train.py first — see "
                f"README for the checkpoint-extraction steps if a resumed run produced "
                f"a corrupted final checkpoint, as happened once already on this project."
            )

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(d))
        except (OSError, FileNotFoundError):
            # Fine-tuning only changes the model's weights, never the
            # tokenizer's vocabulary — so a locally-saved tokenizer file
            # going missing (e.g. a lost Colab session before download) is
            # safe to recover from the public base checkpoint instead; it's
            # byte-identical to what would have been saved locally, not an
            # approximation.
            print("[warn] DistilBertPredictor: local tokenizer files incomplete — "
                  "falling back to the public distilbert-base-uncased tokenizer "
                  "(identical vocabulary, since fine-tuning never touches the tokenizer).")
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._torch = torch
        self._max_len = 128

        n_emotion, n_intent = len(EMOTION_LABELS), len(INTENT_LABELS)

        class _TwoHead(nn.Module):
            """Must exactly match TwoHeadModel's architecture in scripts/train.py —
            load_state_dict below will fail loudly if it doesn't."""

            def __init__(self):
                super().__init__()
                # from_config, not from_pretrained: builds the right-shaped
                # architecture without re-downloading pretrained weights we're
                # about to overwrite entirely via load_state_dict anyway.
                config = AutoConfig.from_pretrained("distilbert-base-uncased")
                self.encoder = AutoModel.from_config(config)
                hidden = self.encoder.config.hidden_size
                self.dropout = nn.Dropout(0.1)
                self.emotion_head = nn.Linear(hidden, n_emotion)
                self.intent_head = nn.Linear(hidden, n_intent)
                self.register_buffer("emotion_weights", torch.zeros(n_emotion))
                self.register_buffer("intent_weights", torch.zeros(n_intent))

            def forward(self, input_ids, attention_mask):
                out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                cls = self.dropout(out.last_hidden_state[:, 0])
                return self.emotion_head(cls), self.intent_head(cls)

        self._model = _TwoHead()
        state_dict = torch.load(model_path, map_location=self.device)
        self._model.load_state_dict(state_dict)   # raises loudly on any shape/key mismatch
        self._model.to(self.device)
        self._model.eval()

        # Optional: temperatures fitted by scripts/calibrate.py from real
        # validation logits. IMPORTANT caveat, confirmed by
        # scripts/check_crisis_calibration.py on this project's real data:
        # a single global temperature is fit against the AGGREGATE
        # validation distribution, which is ~95% two majority classes
        # (emotional_expression, information_seeking) for the intent head.
        # It corrects calibration on those majority classes; it does NOT
        # fix — and on this data actually can't touch — the specific
        # overconfidence seen on the crisis_disclosure minority class or on
        # out-of-distribution input (see the adversarial eval's adv-007,
        # adv-015). calibrated=True here means "a real temperature fit
        # exists and is applied", not "confidence is trustworthy
        # everywhere" — that distinction matters enough to spell out here
        # rather than let the flag imply more than it does.
        self._emotion_temperature = 1.0
        self._intent_temperature = 1.0
        self.calibrated = False
        calib_path = d / "calibration.json"
        if calib_path.exists():
            import json
            entries = {e["head"]: e["temperature"] for e in json.loads(calib_path.read_text())}
            self._emotion_temperature = entries.get("emotion", 1.0)
            self._intent_temperature = entries.get("intent", 1.0)
            self.calibrated = True

    def predict(self, text: str):
        enc = self.tokenizer(text, truncation=True, max_length=self._max_len, return_tensors="pt")
        # Select only the two tensors the model's forward() actually accepts,
        # rather than **-unpacking the whole encoding dict — DistilBERT's
        # tokenizer includes "token_type_ids" by default even though
        # DistilBERT itself never uses it, which a blind **enc crashes on.
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        with self._torch.no_grad():
            emotion_logits, intent_logits = self._model(input_ids=input_ids, attention_mask=attention_mask)
        emotion_probs = self._torch.softmax(emotion_logits / self._emotion_temperature, dim=-1)[0].cpu().tolist()
        intent_probs = self._torch.softmax(intent_logits / self._intent_temperature, dim=-1)[0].cpu().tolist()
        return dict(zip(EMOTION_LABELS, emotion_probs)), dict(zip(INTENT_LABELS, intent_probs))


def load_predictor(kind: str = "stub") -> Predictor:
    if kind == "stub":
        return StubPredictor()
    if kind == "baseline":
        return BaselinePredictor()
    if kind == "distilbert":
        return DistilBertPredictor()
    raise ValueError(f"unknown predictor: {kind}")


# ─────────────────────────────────────────────────────────────────────────
# SERVICE
# ─────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Module 1 — Intent & Emotion Detection",
    version=SCHEMA_VERSION,
    description=(
        "Classification router for the clinical guardrail middleware. "
        "Outputs are advisory signals for Module 2, not clinical judgements. "
        "intent == crisis_disclosure or abstain == true must both escalate."
    ),
)

_predictor = load_predictor(os.getenv("MODULE1_PREDICTOR", "stub"))


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if _predictor.calibrated else "degraded",
        model_version=_predictor.model_version,
        schema_version=SCHEMA_VERSION,
        calibrated=_predictor.calibrated,
        labels={"emotion": EMOTION_LABELS, "intent": INTENT_LABELS},
    )


@app.post("/v1/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest) -> ClassifyResponse:
    started = time.perf_counter()

    emotion_dist, intent_dist = _predictor.predict(req.text)
    abstain, reason = _predictor.should_abstain(emotion_dist, intent_dist)

    top_emotion = max(emotion_dist, key=emotion_dist.get)
    top_intent = max(intent_dist, key=intent_dist.get)

    return ClassifyResponse(
        model_version=_predictor.model_version,
        text_id=req.text_id or str(uuid.uuid4()),
        emotion=Prediction(label=top_emotion, confidence=emotion_dist[top_emotion], distribution=emotion_dist),
        intent=Prediction(label=top_intent, confidence=intent_dist[top_intent], distribution=intent_dist),
        calibrated=_predictor.calibrated,
        abstain=abstain,
        abstain_reason=reason,
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
    )