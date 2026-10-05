"""
main.py — FastAPI application wrapping the existing Agent pipeline.

Run from the REPO ROOT (same assumption every existing script in src/
already makes, since models/retrieval data are loaded via relative paths):

    uvicorn api.main:app --reload

*** DEMO DISCLOSURE ***
This is a PlayStation-style customer-support demo built on the public Kaggle
"Customer Support on Twitter" dataset. It is not connected to any real Sony
system, real customer data, or real account actions. See README.md.

*** PROCESSING MODEL ***
Requests are handled SYNCHRONOUSLY: a POST /tickets call runs the full
pipeline (classification -> retrieval -> reply generation -> safety checks
-> escalation decision) in-request and returns once it's done. There is no
background job queue in this slice -- FastAPI runs these (plain `def`, not
`async def`) route handlers in a thread pool, which allows some request
concurrency, but this is not asynchronous job processing and isn't
described as such anywhere in this codebase.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session, joinedload

from api.database import get_db, init_db
from api.db_models import AuditRecord, EscalationDecision, Ticket
from api.dependencies import get_agent
from api.schemas import ErrorResponse, TicketCreateRequest, TicketResponse

logger = logging.getLogger("hiver_api")
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    get_agent()  # warm the singleton at startup rather than on the first request
    yield


app = FastAPI(
    title="AskPlayStation Support Agent API",
    description=(
        "PlayStation-style customer-support demo API over the public Kaggle "
        "'Customer Support on Twitter' dataset. Not affiliated with or "
        "connected to any real Sony system or real customer data."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


def agent_dependency():
    """Thin wrapper so the Agent singleton is requested via FastAPI's
    Depends(...) mechanism in routes below, not called directly -- this is
    what makes it possible to override with a fake/broken agent in tests
    (see tests/test_api.py's error-handling test) without touching real
    pipeline code."""
    return get_agent()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "demo_disclosure": "PlayStation-style demo over public Kaggle data; not a real Sony system.",
        "processing_model": "synchronous (no background job queue in this slice)",
    }


def _load_ticket_response(db: Session, ticket_id: int) -> TicketResponse | None:
    ticket = (
        db.query(Ticket)
        .options(joinedload(Ticket.escalation_decision))
        .filter(Ticket.id == ticket_id)
        .first()
    )
    if ticket is None:
        return None
    return TicketResponse.model_validate(ticket)


@app.post(
    "/tickets",
    response_model=TicketResponse,
    status_code=201,
    responses={500: {"model": ErrorResponse}},
)
def create_ticket(payload: TicketCreateRequest, db: Session = Depends(get_db), agent=Depends(agent_dependency)):
    prior_context = [turn.model_dump() for turn in payload.prior_context]

    try:
        result = agent.handle(payload.customer_text, prior_context)
    except Exception:
        # Full exception + stack trace goes to the server log only. The API
        # response never includes exception text, file paths, or internals.
        logger.exception("Pipeline failure in POST /tickets (customer_text length=%d)", len(payload.customer_text))
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="pipeline_failure",
                detail="The support agent pipeline failed to process this message. Please try again.",
            ).model_dump(),
        )

    try:
        ticket = Ticket(
            customer_text=result["customer_text"],
            prior_context=result["prior_context"],
            predicted_intent=result["predicted_intent"],
            classifier_confidence=result["classifier_confidence"],
            message_quality=result["message_quality"],
            retrieval_mode=result["retrieval_mode"],
            retrieved_evidence=result["retrieved_evidence"],
            evidence_failure=result["evidence_failure"],
            evidence_failure_reasons=result["evidence_failure_reasons"],
            already_tried_detected=result["already_tried_detected"],
            candidate_reply=result["candidate_reply"],
            polarity_check=result["polarity_check"],
            responsiveness_check=result["responsiveness_check"],
            context_consistency_check=result["context_consistency_check"],
            response_safety_failure=result["response_safety_failure"],
            final_fallback_reason=result.get("final_fallback_reason"),
            generated_reply=result["generated_reply"],
            generation_metadata=result["generation_metadata"],
        )
        db.add(ticket)
        db.flush()  # assigns ticket.id without committing yet

        decision = result["escalation_decision"]
        db.add(EscalationDecision(
            ticket_id=ticket.id,
            action=decision["action"],
            sub_mode=decision.get("sub_mode"),
            escalation_score=decision["escalation_score"],
            escalation_threshold=decision["escalation_threshold"],
            reason_codes=decision["reason_codes"],
            signal_breakdown=decision["signal_breakdown"],
            decision_inputs=decision["decision_inputs"],
        ))
        db.add(AuditRecord(
            ticket_id=ticket.id,
            event_type="ticket_created",
            event_data={"escalation_decision": decision, "predicted_intent": result["predicted_intent"]},
        ))

        db.commit()  # single commit -- ticket, decision, and audit record land together or not at all
    except Exception:
        db.rollback()
        logger.exception("Database failure persisting ticket in POST /tickets")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="persistence_failure",
                detail="The pipeline ran, but the result could not be saved. Please try again.",
            ).model_dump(),
        )

    return _load_ticket_response(db, ticket.id)


@app.get(
    "/tickets/{ticket_id}",
    response_model=TicketResponse,
    responses={404: {"model": ErrorResponse}},
)
def get_ticket(ticket_id: int, db: Session = Depends(get_db)):
    response = _load_ticket_response(db, ticket_id)
    if response is None:
        # Same flat {error, detail} shape as the 500 responses above, rather
        # than FastAPI's default HTTPException shape (which would nest this
        # under a "detail" key) -- keeping error responses uniform.
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(error="not_found", detail=f"No ticket with id {ticket_id}.").model_dump(),
        )
    return response
