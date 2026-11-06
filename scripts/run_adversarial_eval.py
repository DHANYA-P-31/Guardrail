"""
Runs data/adversarial_eval.jsonl against every predictor that's actually
available right now (in-process, no HTTP needed) and prints a side-by-side
comparison, plus per-model detail.

Usage:
    python scripts/run_adversarial_eval.py

Availability is auto-detected:
  - stub: always available.
  - baseline: available once scripts/baseline.py has produced
    artifacts/baseline/*.joblib.
  - distilbert: available once scripts/train.py has produced
    artifacts/distilbert/model.pt AND DistilBertPredictor.predict is
    implemented in service.py (it currently raises NotImplementedError by
    design — this script skips it cleanly rather than crashing until then).

The stub is EXPECTED to fail most cases — it's a keyword heuristic, and
several cases (negation, past tense, implicit crisis) are specifically
designed to break that kind of matching. The comparison table is the point:
it's evidence of improvement (or not) across model stages, not a pass/fail
gate on any one model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from service import load_predictor  # noqa: E402

CASES = ROOT / "data" / "adversarial_eval.jsonl"


def check_case(predictor, case: dict) -> dict:
    emotion, intent = predictor.predict(case["text"])
    abstain, _ = predictor.should_abstain(emotion, intent)
    top_intent = max(intent, key=intent.get)

    expect_intent = case.get("expect_intent")
    if expect_intent == "ABSTAIN":
        passed = abstain
        top1_correct = abstain   # "correct" has no other meaning for this case type
    else:
        top1_correct = (top_intent == expect_intent)
        passed = top1_correct and not abstain

    return {
        "id": case["id"], "category": case["category"], "text": case["text"],
        "passed": passed, "top1_correct": top1_correct,
        "got_intent": top_intent, "expected_intent": expect_intent, "got_abstain": abstain,
    }


def available_predictors() -> dict:
    """Returns {kind: predictor} for every kind that loads successfully right
    now. A missing-artifacts kind is skipped silently (expected — you simply
    haven't reached that stage yet); any other error is surfaced, since that
    likely means something is actually broken."""
    found = {}
    for kind in ("stub", "baseline", "distilbert"):
        try:
            found[kind] = load_predictor(kind)
        except (FileNotFoundError, NotImplementedError) as e:
            print(f"[skip] {kind}: not available yet ({e})")
        except Exception as e:
            print(f"[warn] {kind}: failed to load — {type(e).__name__}: {e}")
    return found


def run_one(kind: str, predictor, cases: list) -> dict:
    results = [check_case(predictor, c) for c in cases]
    n_pass = sum(r["passed"] for r in results)
    n_top1 = sum(r["top1_correct"] for r in results)

    by_category: dict[str, list] = {}
    for r in results:
        by_category.setdefault(r["category"], []).append(r)

    print(f"\n{'=' * 60}")
    print(f"{kind}  ({predictor.model_version})")
    print(f"{'=' * 60}")
    print(f"passed (correct AND not abstaining): {n_pass}/{len(results)}")
    print(f"top-1 correct (ignoring abstain):     {n_top1}/{len(results)}")
    if not predictor.calibrated and n_top1 > n_pass:
        print(f"  -> {n_top1 - n_pass} case(s) had the right answer but abstained. calibrated=False for "
              f"this model, so its confidence numbers are not comparable to ABSTAIN_THRESHOLD in any "
              f"meaningful way yet — a gap between these two numbers is expected and not itself a problem.")
    print()

    for cat, rows in sorted(by_category.items()):
        cat_pass = sum(r["passed"] for r in rows)
        marker = "✓" if cat_pass == len(rows) else "✗"
        print(f"{marker} {cat:20s} {cat_pass}/{len(rows)}")
        for r in rows:
            if not r["passed"]:
                tag = "WRONG" if not r["top1_correct"] else "right-but-abstained"
                print(f"    [{tag}] {r['id']}: got intent={r['got_intent']!r} "
                      f"abstain={r['got_abstain']}, expected {r['expected_intent']!r}")

    return {
        "model_version": predictor.model_version, "calibrated": predictor.calibrated,
        "training_epoch": getattr(predictor, "training_epoch", None),
        "n_pass": n_pass, "n_top1_correct": n_top1, "n_total": len(results), "results": results,
    }


def print_comparison_table(reports: dict, cases: list):
    """Side-by-side pass/fail per case across every model that ran, plus a
    totals row. This is the actual three-way comparison, not just three
    separate single-model print-outs."""
    kinds = list(reports.keys())
    if len(kinds) < 2:
        return   # nothing to compare yet — only one model available

    print(f"\n{'=' * 60}")
    print("COMPARISON")
    print(f"{'=' * 60}")

    id_to_result = {k: {r["id"]: r["passed"] for r in reports[k]["results"]} for k in kinds}

    header = f"{'case':12s} {'category':20s} " + " ".join(f"{k:10s}" for k in kinds)
    print(header)
    for case in cases:
        cid = case["id"]
        row = f"{cid:12s} {case['category']:20s} "
        row += " ".join(f"{'PASS' if id_to_result[k].get(cid) else 'fail':10s}" for k in kinds)
        print(row)

    print()
    for k in kinds:
        n = reports[k]["n_pass"]
        n_top1 = reports[k]["n_top1_correct"]
        total = reports[k]["n_total"]
        note = "" if reports[k]["calibrated"] else "  (uncalibrated — compare top-1, not pass rate)"
        print(f"{k:12s} passed: {n}/{total} ({n/total*100:.0f}%)   top-1 correct: {n_top1}/{total} ({n_top1/total*100:.0f}%){note}")


def main():
    cases = [json.loads(l) for l in open(CASES)]
    predictors = available_predictors()

    if not predictors:
        raise SystemExit("No predictors available at all — this shouldn't happen, 'stub' always loads.")

    reports = {}
    for kind, predictor in predictors.items():
        reports[kind] = run_one(kind, predictor, cases)

    print_comparison_table(reports, cases)

    out = ROOT / "results" / "adversarial_report.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(reports, indent=2))
    print(f"\nFull report (all models): {out}")


if __name__ == "__main__":
    main()