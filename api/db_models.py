"""
db_models.py — SQLAlchemy ORM models.

Three tables, each with a distinct purpose:

- Ticket: one row per POST /tickets call. Stores the full Agent.handle()
  output (minus escalation_decision, which gets its own table) -- this is
  the system-of-record for what the pipeline actually produced.

- EscalationDecision: the escalation_decision sub-object, in its own table
  rather than a JSON blob on Ticket, so it's independently queryable (e.g.
  "all escalations with reason code X"). One-to-one with Ticket in this
  slice; modeled as its own table (rather than inline columns on Ticket) so
  a future human-override flow can add a second decision row per ticket
  without a schema change.

- AuditRecord: an append-only event log keyed by ticket. This slice writes
  exactly one event per ticket ("ticket_created"). Deliberately generic
  (event_type + JSON event_data) so a future human-override endpoint adds a
  new event_type value, not a new column or table.

All JSON-shaped fields (prior_context, retrieved_evidence, reason_codes,
etc.) use SQLAlchemy's JSON type, which both SQLite and PostgreSQL support
natively -- part of why this schema doesn't need to change when the
database URL does.
"""

from datetime import datetime, timezone

from sqlalchemy import (JSON, Boolean, Column, DateTime, Float, ForeignKey,
                         Integer, String, Text)
from sqlalchemy.orm import relationship

from api.database import Base


def utcnow():
    return datetime.now(timezone.utc)


class Ticket(Base):
    __tablename__ = "tickets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    # --- request input ---
    customer_text = Column(Text, nullable=False)
    prior_context = Column(JSON, nullable=False, default=list)  # List[{"role": str, "text": str}]

    # --- intent classification ---
    predicted_intent = Column(String, nullable=False)
    classifier_confidence = Column(Float, nullable=False)
    message_quality = Column(JSON, nullable=False)  # see ReplyGenerator.generate()'s message_quality dict

    # --- retrieval ---
    retrieval_mode = Column(String, nullable=False)  # "global" | "intent_aware"
    retrieved_evidence = Column(JSON, nullable=False, default=list)  # list of evidence dicts

    # --- evidence / safety checks ---
    evidence_failure = Column(Boolean, nullable=False)
    evidence_failure_reasons = Column(JSON, nullable=False, default=list)
    already_tried_detected = Column(Boolean, nullable=False)
    candidate_reply = Column(Text, nullable=False)
    polarity_check = Column(JSON, nullable=False)  # {"passed": bool, "reasons": [...]}
    responsiveness_check = Column(JSON, nullable=False)
    context_consistency_check = Column(JSON, nullable=False)
    response_safety_failure = Column(Boolean, nullable=False)
    final_fallback_reason = Column(String, nullable=True)

    # --- final output ---
    generated_reply = Column(Text, nullable=False)
    generation_metadata = Column(JSON, nullable=False)

    escalation_decision = relationship(
        "EscalationDecision", back_populates="ticket", uselist=False, cascade="all, delete-orphan"
    )
    audit_records = relationship("AuditRecord", back_populates="ticket", cascade="all, delete-orphan")


class EscalationDecision(Base):
    __tablename__ = "escalation_decisions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticket_id = Column(Integer, ForeignKey("tickets.id"), nullable=False, unique=True)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    action = Column(String, nullable=False)  # "auto_handle" | "escalate"
    sub_mode = Column(String, nullable=True)  # "resolve" | "clarify" | None
    escalation_score = Column(Integer, nullable=False)
    escalation_threshold = Column(Integer, nullable=False)
    reason_codes = Column(JSON, nullable=False, default=list)
    signal_breakdown = Column(JSON, nullable=False, default=dict)
    decision_inputs = Column(JSON, nullable=False, default=dict)

    ticket = relationship("Ticket", back_populates="escalation_decision")


class AuditRecord(Base):
    __tablename__ = "audit_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticket_id = Column(Integer, ForeignKey("tickets.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    event_type = Column(String, nullable=False)  # e.g. "ticket_created" (this slice); future: "human_override", etc.
    event_data = Column(JSON, nullable=False, default=dict)

    ticket = relationship("Ticket", back_populates="audit_records")
