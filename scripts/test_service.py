"""Contract tests for service.py. If these break, Module 2 breaks."""

from fastapi.testclient import TestClient

from service import EMOTION_LABELS, HIGH_RISK_INTENT, INTENT_LABELS, BaselinePredictor, app

client = TestClient(app)


def classify(text: str) -> dict:
    r = client.post("/v1/classify", json={"text": text})
    assert r.status_code == 200, r.text
    return r.json()


def test_health_exposes_label_space():
    body = client.get("/health").json()
    assert body["labels"]["emotion"] == EMOTION_LABELS
    assert body["labels"]["intent"] == INTENT_LABELS


def test_response_shape_matches_spec():
    body = classify("I have been feeling very anxious and cannot sleep.")
    for key in ("schema_version", "model_version", "text_id", "emotion", "intent",
                "calibrated", "abstain", "latency_ms"):
        assert key in body
    for head in ("emotion", "intent"):
        for field in ("label", "confidence", "distribution"):
            assert field in body[head]


def test_distributions_are_normalised():
    body = classify("What are the common signs of stress?")
    for head in ("emotion", "intent"):
        total = sum(body[head]["distribution"].values())
        assert abs(total - 1.0) < 0.01, f"{head} sums to {total}"


def test_crisis_language_maps_to_crisis_disclosure_intent():
    body = classify("I don't want to live anymore.")
    assert body["intent"]["label"] == HIGH_RISK_INTENT


def test_no_separate_risk_signals_field():
    body = classify("I want to hurt myself.")
    assert "risk_signals" not in body


def test_unrecognised_input_abstains_rather_than_guessing_normal():
    body = classify("zxcv qwerty asdf")
    assert body["abstain"] is True
    assert body["abstain_reason"]


def test_stub_is_never_marked_calibrated():
    body = classify("hello")
    if "stub" in body["model_version"]:
        assert body["calibrated"] is False


def test_blank_text_rejected():
    assert client.post("/v1/classify", json={"text": "   "}).status_code == 422


def test_baseline_predictor_raises_clearly_when_artifacts_missing():
    # Doesn't require scripts/baseline.py to have run — just confirms the
    # failure mode is a clear, typed error rather than a confusing crash
    # somewhere inside sklearn's loading code.
    try:
        BaselinePredictor(artifacts_dir="/nonexistent/path/for/testing")
        assert False, "should have raised FileNotFoundError"
    except FileNotFoundError as e:
        assert "Run scripts/baseline.py first" in str(e)


# ── Reference client (module1_client.py) ─────────────────────────────────

import httpx
import module1_client as m1c


def test_client_escalates_on_abstain_and_on_crisis_but_not_otherwise():
    assert m1c.should_escalate({"abstain": True, "abstain_reason": "low confidence"})[0] is True
    assert m1c.should_escalate({"abstain": False, "intent": {"label": "crisis_disclosure", "confidence": 0.9}})[0] is True
    esc, why = m1c.should_escalate({"abstain": False, "intent": {"label": "casual_conversation", "confidence": 0.9}})
    assert esc is False and "NOT mean" in why   # the message itself warns this isn't a safety clearance


def test_client_fails_safe_when_service_unreachable():
    resp = m1c.classify("hello", base_url="http://127.0.0.1:1", timeout=0.5)   # nothing listens on port 1
    assert resp["abstain"] is True
    assert "module1_unavailable" in resp["abstain_reason"]
    assert m1c.should_escalate(resp)[0] is True       # unreachable -> escalate, never "normal"


def test_client_fails_safe_on_http_error(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda url, json, timeout: client.post("/v1/classify", json={"text": "   "}))
    resp = m1c.classify("   ")                         # blank text -> 422 from the service
    assert resp["abstain"] is True and m1c.should_escalate(resp)[0] is True


def test_client_rejects_unsupported_schema_major(monkeypatch):
    class Fake:
        def raise_for_status(self): pass
        def json(self): return {"schema_version": "3.0.0", "abstain": False, "intent": {"label": "casual_conversation", "confidence": 0.9}}
    monkeypatch.setattr(httpx, "post", lambda url, json, timeout: Fake())
    resp = m1c.classify("hi")
    assert resp["abstain"] is True and "schema_version" in resp["abstain_reason"]


def test_client_roundtrip_against_real_app_and_flat_contract(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda url, json, timeout: client.post("/v1/classify", json=json))
    resp = m1c.classify("I don't want to live anymore.")
    assert resp["intent"]["label"] == "crisis_disclosure"
    assert m1c.should_escalate(resp)[0] is True
    flat = m1c.to_flat_contract(resp)
    assert set(flat) == {"emotion", "intent", "confidence", "abstain"}
    assert flat["intent"] == "crisis_disclosure"


def test_flat_contract_of_unavailable_response_has_no_intent_and_abstains():
    flat = m1c.to_flat_contract(m1c.classify("x", base_url="http://127.0.0.1:1", timeout=0.3))
    assert flat["intent"] is None and flat["abstain"] is True and flat["confidence"] == 0.0


# ── Artifact lookup (service.py may sit at the project root OR in scripts/) ──

import service as svc


def test_artifacts_found_when_service_py_is_in_scripts_subfolder(tmp_path, monkeypatch):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "artifacts" / "distilbert").mkdir(parents=True)
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.setattr(svc, "_HERE", tmp_path / "scripts")
    monkeypatch.chdir(tmp_path / "elsewhere")
    monkeypatch.delenv("MODULE1_ARTIFACTS_DIR", raising=False)
    assert svc._artifact_dir("distilbert") == tmp_path / "artifacts" / "distilbert"


def test_artifacts_next_to_service_py_win_over_parent(tmp_path, monkeypatch):
    (tmp_path / "flat" / "artifacts" / "baseline").mkdir(parents=True)
    (tmp_path / "artifacts" / "baseline").mkdir(parents=True)
    monkeypatch.setattr(svc, "_HERE", tmp_path / "flat")
    monkeypatch.delenv("MODULE1_ARTIFACTS_DIR", raising=False)
    assert svc._artifact_dir("baseline") == tmp_path / "flat" / "artifacts" / "baseline"


def test_artifacts_env_override_is_strict_even_if_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("MODULE1_ARTIFACTS_DIR", str(tmp_path / "custom"))
    assert svc._artifact_dir("baseline") == tmp_path / "custom" / "baseline"