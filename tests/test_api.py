"""
test_api.py — tests for the FastAPI layer: endpoint behavior, schema
validation, and actual database persistence (not just HTTP-level mocking).

Uses a temporary, isolated SQLite file per test session (never the real
tickets.db), via FastAPI's dependency-override mechanism -- the app code
under test is completely unaware it's running against a test database.

Run with: pytest tests/test_api.py -v
"""

import sys
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.database import Base, get_db
from api.db_models import AuditRecord, EscalationDecision, Ticket
from api.main import agent_dependency, app


@pytest.fixture(scope="session")
def test_db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture(scope="session")
def test_engine(test_db_path):
    engine = create_engine(f"sqlite:///{test_db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    return engine


@pytest.fixture(scope="session")
def TestSessionLocal(test_engine):
    return sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(scope="session", autouse=True)
def override_db_dependency(TestSessionLocal):
    def _get_test_db():
        db = TestSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _get_test_db
    yield
    app.dependency_overrides.clear()


@pytest.fixture(scope="session")
def client():
    # Agent() loads real model/retrieval artifacts -- genuinely exercises the
    # pipeline, not a mock. This is a real integration test, not a unit test
    # with the pipeline stubbed out (see test_error_response_* below for the
    # one test that DOES stub it, specifically to test error handling).
    with TestClient(app) as c:
        yield c


@pytest.fixture
def db_session(TestSessionLocal):
    session = TestSessionLocal()
    yield session
    session.close()


# ---------------------------------------------------------------------------
# Health / demo-disclosure
# ---------------------------------------------------------------------------

def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "demo" in body["demo_disclosure"].lower()
    assert "synchronous" in body["processing_model"].lower()


# ---------------------------------------------------------------------------
# POST /tickets
# ---------------------------------------------------------------------------

def test_create_ticket_success(client):
    resp = client.post("/tickets", json={"customer_text": "my ps4 controller won't charge"})
    assert resp.status_code == 201
    body = resp.json()

    assert isinstance(body["id"], int)
    assert body["customer_text"] == "my ps4 controller won't charge"
    assert body["predicted_intent"]  # non-empty string
    assert 0.0 <= body["classifier_confidence"] <= 1.0
    assert body["retrieval_mode"] in ("global", "intent_aware")
    assert isinstance(body["retrieved_evidence"], list)
    assert "generated_reply" in body and body["generated_reply"]

    decision = body["escalation_decision"]
    assert decision["action"] in ("auto_handle", "escalate")
    assert isinstance(decision["reason_codes"], list)
    assert isinstance(decision["signal_breakdown"], dict)
    assert "decision_inputs" in decision


def test_create_ticket_with_prior_context(client):
    resp = client.post("/tickets", json={
        "customer_text": "I did that already, still not working",
        "prior_context": [
            {"role": "customer", "text": "my download is stuck at 50%"},
            {"role": "brand", "text": "Please try restarting the download"},
        ],
    })
    assert resp.status_code == 201
    body = resp.json()
    assert len(body["prior_context"]) == 2
    assert body["prior_context"][0]["role"] == "customer"
    assert body["prior_context"][1]["role"] == "brand"


def test_create_ticket_rejects_empty_message(client):
    resp = client.post("/tickets", json={"customer_text": ""})
    assert resp.status_code == 422  # Pydantic min_length=1 validation, not a pipeline/DB error


def test_create_ticket_rejects_invalid_role(client):
    resp = client.post("/tickets", json={
        "customer_text": "hello",
        "prior_context": [{"role": "support_agent", "text": "hi"}],  # not "customer" or "brand"
    })
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /tickets/{id}
# ---------------------------------------------------------------------------

def test_get_ticket_matches_create_response(client):
    create_resp = client.post("/tickets", json={"customer_text": "getting error code CE-34878-0"})
    created = create_resp.json()

    get_resp = client.get(f"/tickets/{created['id']}")
    assert get_resp.status_code == 200
    fetched = get_resp.json()

    # The whole point of this test: persisted-and-reloaded data must match
    # what was returned at creation time, field for field, including the
    # nested escalation_decision.
    assert fetched["id"] == created["id"]
    assert fetched["customer_text"] == created["customer_text"]
    assert fetched["predicted_intent"] == created["predicted_intent"]
    assert fetched["classifier_confidence"] == created["classifier_confidence"]
    assert fetched["generated_reply"] == created["generated_reply"]
    assert fetched["escalation_decision"] == created["escalation_decision"]
    assert fetched["retrieved_evidence"] == created["retrieved_evidence"]


def test_get_ticket_not_found_returns_structured_404(client):
    resp = client.get("/tickets/999999")
    assert resp.status_code == 404
    body = resp.json()
    assert body["error"] == "not_found"
    assert "999999" in body["detail"]


# ---------------------------------------------------------------------------
# Database behavior (querying the DB directly, not just through the API)
# ---------------------------------------------------------------------------

def test_ticket_persisted_with_related_rows(client, db_session):
    resp = client.post("/tickets", json={"customer_text": "my account was hacked"})
    ticket_id = resp.json()["id"]

    ticket_row = db_session.query(Ticket).filter(Ticket.id == ticket_id).first()
    assert ticket_row is not None
    assert ticket_row.customer_text == "my account was hacked"

    decision_row = db_session.query(EscalationDecision).filter(EscalationDecision.ticket_id == ticket_id).first()
    assert decision_row is not None
    assert decision_row.action in ("auto_handle", "escalate")
    # "my account was hacked" should trip the high-risk escalation signal
    assert decision_row.action == "escalate"
    assert "HIGH_RISK_INTENT_SIGNAL" in decision_row.reason_codes

    audit_rows = db_session.query(AuditRecord).filter(AuditRecord.ticket_id == ticket_id).all()
    assert len(audit_rows) == 1
    assert audit_rows[0].event_type == "ticket_created"


def test_each_request_creates_a_new_ticket(client, db_session):
    count_before = db_session.query(Ticket).count()
    client.post("/tickets", json={"customer_text": "how do I reset my password"})
    client.post("/tickets", json={"customer_text": "is PSN down right now"})
    count_after = db_session.query(Ticket).count()
    assert count_after == count_before + 2


# ---------------------------------------------------------------------------
# Error handling: no leaked exception internals
# ---------------------------------------------------------------------------

def test_pipeline_failure_returns_safe_error_without_leaking_internals(client, monkeypatch):
    class ExplodingAgent:
        def handle(self, *args, **kwargs):
            raise RuntimeError("secret failure at /home/claude/hiver-support-agent/models/tfidf_logreg_intent_classifier.joblib")

    def fake_agent_dependency():
        return ExplodingAgent()

    app.dependency_overrides[agent_dependency] = fake_agent_dependency
    try:
        resp = client.post("/tickets", json={"customer_text": "this should hit the exploding agent"})
    finally:
        del app.dependency_overrides[agent_dependency]

    assert resp.status_code == 500
    body = resp.json()
    assert body["error"] == "pipeline_failure"
    # The safe, generic message is present...
    assert "failed to process" in body["detail"]
    # ...but nothing about the real exception or internal paths leaked into the response.
    assert "secret failure" not in str(body)
    assert "/home/claude" not in str(body)
    assert ".joblib" not in str(body)
