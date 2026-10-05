"""
schemas.py — Pydantic request/response models.

Every field here was mapped from an ACTUAL live call to Agent.handle()
(inspected directly, not assumed from reading code alone), covering:
  src/agent.py (Agent.handle), src/reply_generator.py (ReplyGenerator.generate,
  which produces most of the fields), src/response_safety.py (the three
  *_check sub-dicts), src/escalation_engine.py (decide(), the
  escalation_decision sub-object).

Nothing here is fabricated or guessed. Fields not present in the real output
are not included.
"""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class PriorContextTurn(BaseModel):
    role: Literal["customer", "brand"]
    text: str


class TicketCreateRequest(BaseModel):
    customer_text: str = Field(..., min_length=1, description="The customer's support message.")
    prior_context: list[PriorContextTurn] = Field(
        default_factory=list, description="Optional prior conversation turns, oldest first."
    )


# ---------------------------------------------------------------------------
# Response sub-objects (mirrors nested dicts in the real pipeline output)
# ---------------------------------------------------------------------------

class CheckResult(BaseModel):
    """Shape of polarity_check / responsiveness_check / context_consistency_check
    in ReplyGenerator.generate()'s output -- confirmed via response_safety.py's
    check_polarity/check_responsiveness/check_context_consistency, each of
    which returns {"passed": bool, "reasons": [...]}."""
    passed: bool
    reasons: list[str]


class MessageQuality(BaseModel):
    word_count: int
    is_very_short: bool
    has_actionable_signal: bool
    is_context_dependent_terse_followup: bool
    is_closing_message: bool
    override_to_low_confidence: bool
    reasons: list[str]


class RetrievedEvidenceItem(BaseModel):
    pair_id: str
    historical_customer_text: str
    historical_brand_reply: str
    similarity_score: float
    predicted_intent: str


class GenerationMetadata(BaseModel):
    method: str
    mock_mode: bool
    llm_provider: Optional[str] = None
    confidence_threshold: float
    weak_similarity_threshold: float


class EscalationDecisionInputs(BaseModel):
    """Mirrors escalation_engine.decide()'s decision_inputs dict exactly."""
    customer_text: str
    predicted_intent: str
    classifier_confidence: float
    evidence_failure: bool
    evidence_failure_reasons: list[str]
    response_safety_failure: bool
    final_fallback_reason: Optional[str] = None
    already_tried_detected: bool
    high_risk_detected: bool
    frustration_detected: bool


class EscalationDecisionOut(BaseModel):
    # from_attributes needed here specifically: on the ORM Ticket object,
    # `escalation_decision` is a related EscalationDecision ORM instance
    # (not a plain dict like the other nested fields below, which come from
    # JSON columns) -- so this nested model needs its own from_attributes
    # to be read via getattr() the same way the parent TicketResponse is.
    model_config = ConfigDict(from_attributes=True)

    action: Literal["auto_handle", "escalate"]
    sub_mode: Optional[Literal["resolve", "clarify"]] = None
    escalation_score: int
    escalation_threshold: int
    reason_codes: list[str]
    signal_breakdown: dict[str, bool]
    decision_inputs: EscalationDecisionInputs


# ---------------------------------------------------------------------------
# Full ticket response
# ---------------------------------------------------------------------------

class TicketResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)  # allows building directly from the ORM object

    id: int
    created_at: datetime

    customer_text: str
    prior_context: list[PriorContextTurn]

    predicted_intent: str
    classifier_confidence: float
    message_quality: MessageQuality

    retrieval_mode: str
    retrieved_evidence: list[RetrievedEvidenceItem]

    evidence_failure: bool
    evidence_failure_reasons: list[str]
    already_tried_detected: bool
    candidate_reply: str
    polarity_check: CheckResult
    responsiveness_check: CheckResult
    context_consistency_check: CheckResult
    response_safety_failure: bool
    final_fallback_reason: Optional[str] = None

    generated_reply: str
    generation_metadata: GenerationMetadata

    escalation_decision: EscalationDecisionOut


# ---------------------------------------------------------------------------
# Error response -- deliberately generic, never includes raw exception text
# or internal paths (see api/main.py's exception handler).
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    error: str
    detail: str
