"""
Merge raw source datasets into the Module 1 schema (2 heads: emotion, intent
— no risk-signals head; crisis severity lives inside intent as
"crisis_disclosure").

Usage:
    python scripts/build_dataset.py

Reads flat from data/ (you populate this by downloading the sources listed
in docs/dataset-sources.md — this script does not fetch anything itself).
Expected filenames:

    data/dair-ai-train.parquet
    data/dair-ai-validation.parquet
    data/dair-ai-test.parquet
    data/dreaddit.csv
    data/goemotions_1.csv
    data/goemotions_2.csv
    data/goemotions_3.csv
    data/counsel-chat.parquet
    data/MHDialog-train.parquet
    data/MHDialog-validation.parquet
    data/MHDialog-test.parquet
    data/dailydialogtrain.csv
    data/dailydialogtest.csv
    data/dailydialogvalidation.csv

Writes data/processed/{train,val,test}.jsonl plus
data/processed/split_manifest.json (test-set hash, frozen once written).

IMPORTANT: column names below are written from public documentation / direct
inspection of each dataset's release format. Verify with scripts/inspect_raw.py
before trusting a full run.

Each loader returns a DataFrame with exactly: text, emotion, intent,
emotion_reliable, intent_reliable.

emotion_reliable / intent_reliable: most sources only have genuine per-example
annotation for ONE of the two heads — the other gets a blanket default
("neutral" / "emotional_expression") because the source simply doesn't carry
that signal. Training the intent head on 220k rows that all say
"emotional_expression" because that's the fallback, not because anyone
labelled them that way, teaches the model "always guess this label" — worse
than having no data for those rows. scripts/baseline.py and scripts/train.py
filter each head's training data to rows where that head's flag is True.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from service import EMOTION_LABELS, INTENT_LABELS  # noqa: E402

RAW = Path(__file__).resolve().parent.parent / "data"  # flat — files sit directly here
OUT = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
VAL_FRAC = 0.15
TEST_FRAC = 0.15


def _clean_text(t: str) -> str:
    t = str(t).strip()
    t = re.sub(r"http\S+|www\.\S+", "", t)
    t = re.sub(r"/?u/\w+|/?r/\w+", "", t)          # reddit user/subreddit refs
    t = re.sub(r"@\w+", "", t)                     # @handles
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ── Source loaders ──────────────────────────────────────────────────────
# RAW is flat — every file sits directly in data/, not in per-source
# subfolders. Loaders match by filename pattern (case-insensitive) against
# whatever's actually in that directory.

def _find_files(stems: list[str]) -> list[Path]:
    """All files in the flat data/ dir whose name contains any of `stems`
    (case-insensitive), parquet/csv/tsv only."""
    if not RAW.exists():
        return []
    hits = []
    for f in RAW.iterdir():
        if f.suffix not in (".parquet", ".csv", ".tsv"):
            continue
        low = f.name.lower()
        if any(stem in low for stem in stems):
            hits.append(f)
    return sorted(hits)


def _find_raw_file(stems: list[str]) -> Optional[Path]:
    hits = _find_files(stems)
    return hits[0] if hits else None


def _read_any(f: Path) -> pd.DataFrame:
    if f.suffix == ".parquet":
        return pd.read_parquet(f)
    if f.suffix == ".tsv":
        return pd.read_csv(f, sep="\t")
    return pd.read_csv(f)


def load_dair_emotion(_unused: Path) -> Optional[pd.DataFrame]:
    """dair-ai/emotion. HF export is parquet with columns: text, label (int 0-5).
    0 sadness, 1 joy, 2 love, 3 anger, 4 fear, 5 surprise.
    Pools train/validation/test.parquet — we do our own group split later."""
    files = _find_files(["dair-ai", "dair_ai"])
    if not files:
        return None
    df = pd.concat([_read_any(f) for f in files], ignore_index=True)
    print(f"       dair-ai: combined {len(files)} file(s) -> {len(df)} rows")

    int_to_ours = {0: "sadness", 1: "joy", 2: "love_care", 3: "anger", 4: "fear", 5: "surprise"}
    label_col = "label" if "label" in df.columns else df.columns[-1]
    if pd.api.types.is_integer_dtype(df[label_col]):
        df["emotion"] = df[label_col].map(int_to_ours)
    else:
        name_map = {"love": "love_care", "surprise": "surprise", "sadness": "sadness",
                    "joy": "joy", "anger": "anger", "fear": "fear"}
        df["emotion"] = df[label_col].astype(str).str.lower().map(name_map)

    unmapped = df["emotion"].isna().sum()
    if unmapped:
        print(f"[warn] dair-ai: {unmapped} rows with unrecognised label, dropping")
    df = df.dropna(subset=["emotion"]).copy()
    df["text"] = df["text"].map(_clean_text)
    df["intent"] = "emotional_expression"       # blanket default — no real intent signal in this source
    df["emotion_reliable"] = True
    df["intent_reliable"] = False
    return df[["text", "emotion", "intent", "emotion_reliable", "intent_reliable"]]


_GOEMOTIONS_28 = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]

_GOEMOTIONS_TO_OURS = {
    "nervousness": "fear", "fear": "fear",
    "sadness": "sadness", "grief": "sadness", "disappointment": "sadness",
    "annoyance": "anger", "disgust": "anger", "disapproval": "anger", "anger": "anger",
    "confusion": "confusion", "curiosity": "confusion",
    "remorse": "guilt_remorse", "embarrassment": "guilt_remorse",
    "joy": "joy", "amusement": "joy", "excitement": "joy",
    "love": "love_care", "caring": "love_care",
    "gratitude": "hope_optimism", "optimism": "hope_optimism", "relief": "hope_optimism",
    "surprise": "surprise",
    "neutral": "neutral", "approval": "neutral", "admiration": "neutral",
    # dropped (no clean home): desire, pride, realization
}


def load_goemotions(_unused: Path) -> Optional[pd.DataFrame]:
    """GoEmotions, sharded as goemotions_1/2/3.csv."""
    files = _find_files(["goemotions"])
    if not files:
        return None
    df = pd.concat([_read_any(f) for f in files], ignore_index=True)
    print(f"       goemotions: combined {len(files)} shard(s) -> {len(df)} rows")

    text_col = "text" if "text" in df.columns else df.columns[0]
    named_cols = [c for c in _GOEMOTIONS_28 if c in df.columns]

    rows_out = []
    if named_cols:
        for _, row in df.iterrows():
            active = [c for c in named_cols if row[c] == 1]
            mapped = next((_GOEMOTIONS_TO_OURS[c] for c in active if c in _GOEMOTIONS_TO_OURS), None)
            if mapped is None:
                continue
            rows_out.append({"text": row[text_col], "emotion": mapped})
    elif "labels" in df.columns:
        import ast
        for _, row in df.iterrows():
            raw = row["labels"]
            ids = raw if isinstance(raw, (list, tuple)) else ast.literal_eval(str(raw))
            names = [_GOEMOTIONS_28[i] for i in ids if i < len(_GOEMOTIONS_28)]
            mapped = next((_GOEMOTIONS_TO_OURS[n] for n in names if n in _GOEMOTIONS_TO_OURS), None)
            if mapped is None:
                continue
            rows_out.append({"text": row[text_col], "emotion": mapped})
    else:
        raise KeyError(
            f"GoEmotions: recognised neither named one-hot columns nor a 'labels' "
            f"column. Actual columns: {df.columns.tolist()}."
        )

    out = pd.DataFrame(rows_out)
    out["text"] = out["text"].map(_clean_text)
    out["intent"] = "emotional_expression"
    out["emotion_reliable"] = True
    out["intent_reliable"] = False
    return out[["text", "emotion", "intent", "emotion_reliable", "intent_reliable"]]


def load_dreaddit(_unused: Path) -> Optional[pd.DataFrame]:
    """Dreaddit. Columns: text,label (1=stress,0=non-stress)."""
    f = _find_raw_file(["dreaddit"])
    if f is None:
        return None
    df = _read_any(f).copy()
    text_col = "text" if "text" in df.columns else "sentence"
    if "label" not in df.columns:
        raise KeyError(f"Unrecognised Dreaddit columns: {df.columns.tolist()}")
    df["emotion"] = df["label"].map({1: "fear", 0: "neutral"})
    df["text"] = df[text_col].map(_clean_text)
    df["intent"] = "problem_description"
    df["emotion_reliable"] = True
    df["intent_reliable"] = False
    return df[["text", "emotion", "intent", "emotion_reliable", "intent_reliable"]]


# Pattern rules for a direct-ask sentence (Counsel-Chat titles, DailyDialog
# question-act utterances). Checked in order; first match wins.
_DIRECT_ASK_RULES = [
    (r"^(where|who) (can|do) i (find|get|go)", "help_seeking"),
    (r"\b(find|recommend)\w*\b.{0,20}\b(therapist|counsel(l)?or|psychiatrist)\b", "help_seeking"),
    (r"^(how do i|how can i|how should i|what should i|what can i) do\b", "advice_seeking"),
    (r"^should i\b", "advice_seeking"),
    (r"^(what is|what are|why do(es)?|why is|why am i|what causes|is it normal)\b", "information_seeking"),
    (r"^(how do i|how can i)\b", "information_seeking"),
    (r"^(do i|am i|can i)\b", "problem_description"),
]


def _classify_direct_ask(text: str, default: str) -> str:
    low = str(text).strip().lower()
    for pattern, intent in _DIRECT_ASK_RULES:
        if re.search(pattern, low):
            return intent
    return default


# Checked BEFORE any other classification. Previously nothing in this entire
# pipeline ever produced "casual_conversation" — not one loader assigned it,
# so the intent head had ZERO training examples for this class (worse than
# crisis_disclosure's 27: this one didn't even show up as a 0-support row in
# the classification report, sklearn just omits a class absent from y_true
# entirely). DailyDialog is a general chit-chat corpus, so it's the obvious
# place to recover real examples rather than inventing synthetic ones.
# Unambiguous discourse markers — safe to match as a start-anchored prefix
# regardless of what follows, since real content essentially never pivots
# from "hi"/"thanks"/"bye" into something serious using that exact opener.
_CASUAL_PATTERN = re.compile(
    r"^(hi|hello|hey|good (morning|afternoon|evening))\b"
    r"|^(how are you|how('s| is) it going|what'?s up)\b"
    r"|^(thanks|thank you|appreciate it)\b"
    r"|^(bye|goodbye|see you|take care|good night)\b"
)

# Short acknowledgment words ("sure", "fine", "ok", "great"...) are NOT safe
# as an unrestricted prefix match: tested and confirmed real false positives
# like "Sure, I guess I should talk to someone about hurting myself." and
# "Fine, I will just deal with this alone." — both would wrongly become
# casual_conversation on a prefix-only check. These words only count as
# casual when the ENTIRE utterance is short (<=4 words): a genuine
# acknowledgment is brief; a disclosure that happens to open with "Sure,"
# is not, and length is the distinguishing signal a bare prefix match lacks.
_CASUAL_ACK_WORDS = re.compile(
    r"^(ok|okay|sure|sounds good|great|cool|nice|alright|fine|no problem|sure thing)\b"
)


def _classify_casual_first(text: str, fallback_classifier, default: str) -> str:
    low = str(text).strip().lower()
    if _CASUAL_PATTERN.search(low):
        return "casual_conversation"
    if _CASUAL_ACK_WORDS.search(low) and len(low.split()) <= 4:
        return "casual_conversation"
    return fallback_classifier(text, default)


def load_counsel_chat(_unused: Path) -> Optional[pd.DataFrame]:
    """Counsel-Chat. Columns: questionID, questionTitle, questionText,
    questionLink, topic, therapistInfo, therapistURL, answerText, upvotes, views.

    topic=="self-harm" or "suicide" -> crisis_disclosure, but confirmed on the
    real file: self-harm has exactly 1 row, suicide has 0. Does NOT meaningfully
    solve the crisis_disclosure gap on its own."""
    f = _find_raw_file(["counsel"])
    if f is None:
        return None
    df = _read_any(f).copy()

    if "questionTitle" not in df.columns:
        raise KeyError(f"Expected 'questionTitle' column, got: {df.columns.tolist()}")

    df["text"] = df["questionTitle"].map(_clean_text)
    df["emotion"] = "neutral"

    topic = df["topic"].astype(str).str.lower() if "topic" in df.columns else ""
    is_crisis_topic = (topic == "self-harm") | (topic == "suicide")

    df["intent"] = df["questionTitle"].map(lambda t: _classify_direct_ask(t, "problem_description"))
    df.loc[is_crisis_topic, "intent"] = "crisis_disclosure"

    n_crisis = int(is_crisis_topic.sum())
    if n_crisis:
        print(f"       counsel_chat: {n_crisis} row(s) tagged crisis_disclosure via topic "
              f"(too few to train on alone)")

    df["emotion_reliable"] = False      # no real emotion signal in this source
    df["intent_reliable"] = True        # content-derived (title pattern + topic), not a blanket default
    return df[["text", "emotion", "intent", "emotion_reliable", "intent_reliable"]]


# MHDialog's 8 Dialog Intent values -> our 8-intent taxonomy. Three values
# (Recovery, Avoidance, Adversarial) have no clean home and are dropped
# rather than force-mapped.
_MHDIALOG_INTENT_MAP = {
    "Emotional Venting": "emotional_expression",
    "Validation-Seeking": "emotional_support",
    "Explicit Help-seeking": "help_seeking",
    "Implicit Help-seeking": "problem_description",
    "CrisisEscalation": "crisis_disclosure",
    # dropped: "Recovery", "Avoidance", "Adversarial"
}

# Complete Level (risk severity) value set, confirmed on the real 700-row
# train file (value_counts: Not Related 398, Moderate 101, Minor 84, Unsure 70,
# Severe 29, No 18). ANY value outside this dict raises loudly — do not
# silently guess at severity ordering for a safety-relevant field.
#
# "Severe" -> crisis override is unambiguous. "No" is NOT confidently
# resolved: it could mean "no risk" (parallel to Not Related -> non-crisis,
# the mapping used here) or something else in the original annotation
# scheme that happens to share a name with the obvious reading. Verify by
# sampling a few real rows where Level == "No" (check their Concern Type
# and Dialogue content) before trusting this on a safety-relevant field —
# the loader prints a warning every time it encounters one as a reminder.
_MHDIALOG_LEVEL_IS_CRISIS = {
    "Not Related": False,
    "Unsure": False,
    "Minor": False,
    "Moderate": False,
    "Severe": True,
    "No": False,   # UNVERIFIED ASSUMPTION — see comment above, confirm before trusting
}


def load_mhdialog(_unused: Path) -> Optional[pd.DataFrame]:
    """MHDialog. Columns: Dialogue (JSON string, list of {round, user, supporter}),
    Dialog Intent (8-class), Concern Type (7-class, unused here), Level (6-class
    risk severity, unused for emotion — only used to sanity-check/override intent).

    One row per DIALOGUE (not per turn): all "user" turns concatenated. This
    avoids applying one dialogue-level label to every individual turn as if
    each turn independently earned it — a defensible approximation, but still
    an approximation; note it in the report. No emotion annotation in this
    source at all."""
    files = _find_files(["mhdialog"])
    if not files:
        return None

    rows_out = []
    n_dropped_intent = 0
    n_missing_dialogue = 0     # empty / NaN-as-string — expected, not a data problem
    n_malformed_dialogue = 0   # genuine JSON syntax errors — worth a closer look if this grows
    unknown_levels = set()

    for f in files:
        df = _read_any(f)
        for _, row in df.iterrows():
            raw_dialogue = row["Dialogue"]
            if not isinstance(raw_dialogue, str) or not raw_dialogue.strip() or raw_dialogue.strip().lower() == "nan":
                n_missing_dialogue += 1
                continue

            try:
                turns = json.loads(raw_dialogue)
            except (json.JSONDecodeError, TypeError):
                n_malformed_dialogue += 1
                continue

            user_text = " ".join(t.get("user", "") for t in turns if t.get("user"))
            if not user_text.strip():
                continue

            # Level is checked FIRST and independently of Dialog Intent. This
            # ordering matters: the whole point of the risk-severity override
            # is to catch a crisis even when the surface Dialog Intent doesn't
            # read as one (e.g. coded "Adversarial" while the underlying risk
            # is Severe). Checking Dialog Intent first and dropping unmapped
            # rows before Level is ever consulted would silently discard
            # exactly the cases this override exists to catch — that was a
            # real bug in an earlier version of this loader, not a hypothetical.
            level = row.get("Level")
            if level not in _MHDIALOG_LEVEL_IS_CRISIS:
                unknown_levels.add(level)
                continue   # don't guess on an unrecognised severity value

            if _MHDIALOG_LEVEL_IS_CRISIS[level]:
                intent = "crisis_disclosure"   # overrides regardless of Dialog Intent, including the dropped categories
            else:
                intent = _MHDIALOG_INTENT_MAP.get(row["Dialog Intent"])
                if intent is None:
                    n_dropped_intent += 1
                    continue

            rows_out.append({"text": _clean_text(user_text), "intent": intent})

    if unknown_levels:
        raise ValueError(
            f"MHDialog: unrecognised Level value(s) {sorted(unknown_levels)} — "
            f"update _MHDIALOG_LEVEL_IS_CRISIS in build_dataset.py with these "
            f"before running for real. Refusing to guess on a safety-relevant field."
        )

    if n_missing_dialogue:
        print(f"       mhdialog: {n_missing_dialogue} row(s) had empty/missing Dialogue — skipped (expected, not a parsing bug)")
    if n_malformed_dialogue:
        print(f"       mhdialog: {n_malformed_dialogue} row(s) had genuinely malformed Dialogue JSON — skipped "
              f"(worth a manual look if this number is large; likely an unescaped quote in generated text)")
    if n_dropped_intent:
        print(f"       mhdialog: dropped {n_dropped_intent} row(s) with Dialog Intent "
              f"in {{Recovery, Avoidance, Adversarial}} — no clean home in our taxonomy")

    out = pd.DataFrame(rows_out)
    if out.empty:
        return out
    out["emotion"] = "neutral"           # no emotion annotation in this source
    out["emotion_reliable"] = False
    out["intent_reliable"] = True        # dialogue-level expert annotation, incl. risk override
    return out[["text", "emotion", "intent", "emotion_reliable", "intent_reliable"]]


# Standard DailyDialog emotion codes (0-6) -> our 10-label taxonomy.
_DAILYDIALOG_EMOTION_MAP = {
    0: "neutral", 1: "anger", 2: "anger",   # 2=disgust, no dedicated bucket -> anger
    3: "fear", 4: "joy", 5: "sadness", 6: "surprise",
}

# Standard DailyDialog dialogue-act codes. 1=inform, 2=question, 3=directive,
# 4=commissive. directive/commissive describe general conversational
# structure ("I suggest...", "Let's go") rather than support-seeking
# behaviour — forcing them onto our intent taxonomy would repeat the exact
# mistake fixed in Counsel-Chat, so their intent is left unreliable instead.


def _parse_quoted_spans(raw: str) -> list[str]:
    """Extracts each single- or double-quoted span from the raw cell text, in
    order. Deliberately NOT ast.literal_eval: tested against the real file and
    confirmed the quoted strings are NOT comma-separated, so literal_eval
    silently concatenates them into ONE string (verified bug, not theoretical)."""
    pattern = r"'((?:[^'\\]|\\.)*)'|\"((?:[^\"\\]|\\.)*)\""
    spans = re.findall(pattern, raw)
    return [a if a else b for a, b in spans]


def _parse_int_codes(raw: str) -> list[int]:
    return [int(x) for x in re.findall(r"\d+", str(raw))]


def load_dailydialog(_unused: Path) -> Optional[pd.DataFrame]:
    """DailyDialog. Columns: dialog, act, emotion — each a stringified
    per-utterance array. One row per UTTERANCE (not per dialogue): act and
    emotion are genuinely per-utterance in this dataset, unlike MHDialog.

    KNOWN DATA GLITCH, confirmed on the real file: some act/emotion strings
    have two digits run together with no space (e.g. "00" instead of "0 0"),
    which silently undercounts by one per occurrence. Any dialogue where the
    parsed code count doesn't match the parsed utterance count is skipped
    entirely rather than guessing which utterance lost its label — a
    misaligned label is worse than a missing row."""
    f = _find_raw_file(["dailydialog"]) if False else None
    files = _find_files(["dailydialog"])
    if not files:
        return None

    rows_out = []
    n_skipped_mismatch = 0

    for f in files:
        df = _read_any(f)
        for _, row in df.iterrows():
            utterances = _parse_quoted_spans(str(row["dialog"]))
            acts = _parse_int_codes(row["act"])
            emotions = _parse_int_codes(row["emotion"])

            if not (len(utterances) == len(acts) == len(emotions)):
                n_skipped_mismatch += 1
                continue

            for text, act, emo in zip(utterances, acts, emotions):
                text = _clean_text(text)
                if not text:
                    continue
                emotion_label = _DAILYDIALOG_EMOTION_MAP.get(emo, "neutral")

                if act == 1:          # inform
                    intent = _classify_casual_first(text, lambda t, d: d, "emotional_expression")
                    intent_reliable = True
                elif act == 2:        # question
                    intent = _classify_casual_first(text, _classify_direct_ask, "information_seeking")
                    intent_reliable = True
                else:                 # directive (3) / commissive (4) — no clean home
                    intent, intent_reliable = "emotional_expression", False

                rows_out.append({
                    "text": text, "emotion": emotion_label, "intent": intent,
                    "emotion_reliable": True, "intent_reliable": intent_reliable,
                })

    if n_skipped_mismatch:
        print(f"       dailydialog: skipped {n_skipped_mismatch} dialogue(s) where "
              f"utterance/act/emotion counts didn't align (known '00'-type parsing glitch)")

    return pd.DataFrame(rows_out)


LOADERS = {
    "emotion": load_dair_emotion,
    "goemotions": load_goemotions,
    "dreaddit": load_dreaddit,
    "counsel_chat": load_counsel_chat,
    "mhdialog": load_mhdialog,
    "dailydialog": load_dailydialog,
}


def build() -> pd.DataFrame:
    frames = []
    for name, loader in LOADERS.items():
        try:
            df = loader(RAW)
        except (KeyError, ValueError) as e:
            print(f"[fail] {name}: {e}")
            continue
        if df is None or df.empty:
            print(f"[skip] {name}: no matching file found in data/, or nothing survived loading")
            continue
        df["source"] = name
        frames.append(df)
        print(f"[ok]   {name}: {len(df)} rows")

    if not frames:
        raise SystemExit(
            "No source data found. Expected files directly under data/ — see "
            "the module docstring for exact filenames — or run with --synthetic "
            "for a smoke test."
        )

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.dropna(subset=["text"])
    merged = merged[merged["text"].str.len() > 0]
    merged = merged.drop_duplicates(subset=["text"])

    bad_emotion = ~merged["emotion"].isin(EMOTION_LABELS)
    bad_intent = ~merged["intent"].isin(INTENT_LABELS)
    if bad_emotion.any() or bad_intent.any():
        raise ValueError(
            f"{bad_emotion.sum()} rows with unmapped emotion, "
            f"{bad_intent.sum()} with unmapped intent — fix the loader mapping."
        )

    n_intent_reliable = int(merged["intent_reliable"].sum())
    print(f"\n{n_intent_reliable}/{len(merged)} rows have content-derived intent labels "
          f"({n_intent_reliable / len(merged) * 100:.1f}%) — scripts/baseline.py and "
          f"scripts/train.py train the intent head on this subset only.")

    return merged.reset_index(drop=True)


def group_split(df: pd.DataFrame):
    """Stratified split, with an unstratified fallback when a class is too
    rare to stratify on (sklearn needs >=2 rows per class per split)."""
    from sklearn.model_selection import train_test_split

    def _safe_strat(frame: pd.DataFrame):
        counts = frame["emotion"].value_counts()
        if frame["emotion"].nunique() <= 1:
            return None
        if (counts < 2).any():
            rare = counts[counts < 2].index.tolist()
            print(f"[warn] split: classes too rare to stratify on ({rare}), "
                  f"falling back to unstratified split for this cut")
            return None
        return frame["emotion"]

    train, rest = train_test_split(df, test_size=VAL_FRAC + TEST_FRAC, random_state=SEED, stratify=_safe_strat(df))
    val, test = train_test_split(
        rest, test_size=TEST_FRAC / (VAL_FRAC + TEST_FRAC), random_state=SEED, stratify=_safe_strat(rest)
    )
    return train, val, test


def write_splits(train, val, test):
    for name, split in (("train", train), ("val", val), ("test", test)):
        path = OUT / f"{name}.jsonl"
        with open(path, "w") as fh:
            for _, row in split.iterrows():
                fh.write(json.dumps({
                    "text": row["text"], "emotion": row["emotion"], "intent": row["intent"],
                    "emotion_reliable": bool(row.get("emotion_reliable", True)),
                    "intent_reliable": bool(row.get("intent_reliable", True)),
                    "source": row.get("source", "synthetic"),
                }) + "\n")
        print(f"wrote {path} ({len(split)} rows)")

    test_hash = hashlib.sha256("\n".join(sorted(test["text"].tolist())).encode()).hexdigest()
    manifest = {
        "seed": SEED, "val_frac": VAL_FRAC, "test_frac": TEST_FRAC,
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "test_set_sha256": test_hash,
        "note": "Do not look at test.jsonl content until the final reported run.",
    }
    (OUT / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"test set hash: {test_hash[:16]}…")


def make_synthetic(n_per_class: int = 8) -> pd.DataFrame:
    """Small, clearly-fake dataset so the pipeline can be proven to run
    without real data present. NEVER use these numbers as reported results."""
    import itertools
    samples = {
        "joy": ["Today was actually a really good day.", "I'm so happy right now."],
        "sadness": ["I've been feeling really down since the breakup.", "Everything feels grey lately."],
        "anger": ["I'm so frustrated with how this was handled.", "It makes me furious every time."],
        "fear": ["I can't stop worrying about the exam tomorrow.", "My heart is racing and I feel panicky."],
        "love_care": ["I really care about how they're doing.", "I feel so much love for my family."],
        "surprise": ["I did not expect that at all.", "Wow, that came out of nowhere."],
        "confusion": ["I don't understand what's happening.", "I'm not sure what to make of this."],
        "guilt_remorse": ["I feel terrible about what I said.", "I regret how I handled that."],
        "hope_optimism": ["I'm hopeful things will get better.", "I think this year will be good."],
        "neutral": ["The meeting is at 3pm.", "I need to submit the form by Friday."],
    }
    rows = []
    for emo, texts in samples.items():
        for t in itertools.islice(itertools.cycle(texts), n_per_class):
            intent = "crisis_disclosure" if emo == "sadness" else "emotional_expression"
            rows.append({
                "text": t, "emotion": emo, "intent": intent,
                "emotion_reliable": True, "intent_reliable": True, "source": "synthetic",
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", action="store_true",
                    help="Use a tiny fabricated dataset instead of data/. "
                         "For proving the pipeline runs only — never for real results.")
    args = p.parse_args()

    df = make_synthetic() if args.synthetic else build()
    train, val, test = group_split(df)
    write_splits(train, val, test)