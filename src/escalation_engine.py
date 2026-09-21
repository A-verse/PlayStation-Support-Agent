"""
escalation_engine.py — auto-handle vs. escalate decision, built on top of
ReplyGenerator's output. Kept as its own module/stage (not baked into
reply_generator.py) so the decision logic is independently auditable,
testable, and swappable.

===============================================================================
DESIGN: an explicit, additive point system -- not a black box
===============================================================================
Every signal below contributes a documented number of points to an
escalation score; ESCALATE if score >= ESCALATION_THRESHOLD (2), else
AUTO_HANDLE. This is deliberately simple and inspectable (a reviewer can
trace exactly why any decision was made) rather than a learned model, per
the project's "explainability over sophistication" principle.

Signals and points (all logged per-decision, not just the total):
  HIGH_RISK_INTENT_SIGNAL         +3   account compromise/ban, financial
                                        dispute, RMA-hardware keywords
                                        (reuses the same conservative
                                        criteria established during golden-
                                        set labeling)
  INSUFFICIENT_EVIDENCE_ON_REAL_INTENT +1   evidence_failure is True AND the
                                        reason is NOT solely "the message
                                        itself is unclear/low-quality" --
                                        i.e. we understood the topic but
                                        can't safely ground an answer
  LOW_CLASSIFIER_CONFIDENCE       +1   confidence < CONFIDENCE_THRESHOLD
                                        (0.6, from the reply-generation stage)
  SAFETY_CHECK_FALLBACK           +1   response_safety_failure is True
  EXHAUSTED_TROUBLESHOOTING_SIGNAL +1  already_tried_detected is True --
                                        a SIGNAL per the approved policy,
                                        not an absolute rule on its own
  FRUSTRATION_DETECTED            +0   logged for observability ONLY.
                                        Per explicit policy: "customer
                                        frustration alone must not trigger
                                        escalation." Never contributes points.
  UNCLEAR_INTENT_OR_MESSAGE_QUALITY +0  a genuinely vague/low-quality message
                                        is handled by a safe clarifying
                                        question (already the reply
                                        generator's fallback), not escalated
                                        to a human by itself.

This means: a single exhausted-troubleshooting report, or a single safety
fallback, or a single low-confidence classification, or plain frustration,
each alone stays at AUTO_HANDLE (with the safe fallback reply already built
into the reply generator) -- escalation requires either one high-risk
signal, or at least two of the weaker signals compounding.
"""

import re

ESCALATION_THRESHOLD = 2

FRUSTRATION_RE = re.compile(
    r"\b(wtf|ridiculous|unacceptable|furious|angry|fed up|sick of|worst|useless|"
    r"terrible|awful|pathetic|joke|scam|garbage)\b|!!+|\?\?+", re.I
)

# Same conservative criteria approved during golden-set labeling (decision
# log #12 in that stage) -- reused here rather than re-invented.
HIGH_RISK_RE = re.compile(
    r"\b(hack(ed|ing)?|stolen|scam(mer|med)?|compromised|banned?|suspend(ed|ing)?|"
    r"refund|dispute|charge\s?back|unauthoriz(ed)?\s+(charge|purchase)|fraud(ulent)?|"
    r"double charged|charged twice|gotten took|"
    r"broken|dead|crack|blue light|disc.{0,15}(stuck|won.?t eject)|bricked|rma|repair)\b", re.I
)


def detect_frustration(customer_text):
    return bool(FRUSTRATION_RE.search(customer_text))


def detect_high_risk(customer_text, prior_context):
    combined = customer_text + " " + " ".join(c["text"] for c in (prior_context or []))
    return bool(HIGH_RISK_RE.search(combined))


def decide(pipeline_result: dict) -> dict:
    """pipeline_result is ReplyGenerator.generate()'s output dict.
    Returns a decision dict with the action, reason codes, per-signal
    breakdown, and the exact inputs the decision was made from -- the full
    audit trail, not just the final verdict."""
    customer_text = pipeline_result["customer_text"]
    prior_context = pipeline_result.get("prior_context", [])

    high_risk = detect_high_risk(customer_text, prior_context)
    frustration = detect_frustration(customer_text)

    evidence_failure = pipeline_result["evidence_failure"]
    evidence_reasons = pipeline_result.get("evidence_failure_reasons", [])
    # "Insufficient evidence on a REAL intent" excludes cases where the ONLY
    # reason is that the message itself is unclear/low-quality -- those are
    # handled by a safe clarifying question, not counted toward escalation.
    non_quality_reasons = [r for r in evidence_reasons if r not in ("unclear_intent",) and not r.startswith("message_quality_guard")]
    insufficient_evidence_on_real_intent = evidence_failure and len(non_quality_reasons) > 0

    low_confidence = pipeline_result["classifier_confidence"] < 0.6
    safety_fallback = pipeline_result.get("response_safety_failure", False)
    exhausted = pipeline_result.get("already_tried_detected", False)

    signals = {
        "HIGH_RISK_INTENT_SIGNAL": (high_risk, 3),
        "INSUFFICIENT_EVIDENCE_ON_REAL_INTENT": (insufficient_evidence_on_real_intent, 1),
        "LOW_CLASSIFIER_CONFIDENCE": (low_confidence, 1),
        "SAFETY_CHECK_FALLBACK": (safety_fallback, 1),
        "EXHAUSTED_TROUBLESHOOTING_SIGNAL": (exhausted, 1),
        "FRUSTRATION_DETECTED": (frustration, 0),  # never contributes -- observability only
    }

    score = sum(points for fired, points in signals.values() if fired)
    reason_codes = [name for name, (fired, _) in signals.items() if fired]

    action = "escalate" if score >= ESCALATION_THRESHOLD else "auto_handle"

    # Sub-mode for auto_handle: was this a confident grounded answer, or a
    # safe clarifying question? Matters for the frontend/eval to distinguish.
    if action == "auto_handle":
        sub_mode = "clarify" if (evidence_failure or safety_fallback) else "resolve"
    else:
        sub_mode = None

    return {
        "action": action,
        "sub_mode": sub_mode,
        "escalation_score": score,
        "escalation_threshold": ESCALATION_THRESHOLD,
        "reason_codes": reason_codes,
        "signal_breakdown": {name: fired for name, (fired, _) in signals.items()},
        "decision_inputs": {
            "customer_text": customer_text,
            "predicted_intent": pipeline_result["predicted_intent"],
            "classifier_confidence": pipeline_result["classifier_confidence"],
            "evidence_failure": evidence_failure,
            "evidence_failure_reasons": evidence_reasons,
            "response_safety_failure": safety_fallback,
            "final_fallback_reason": pipeline_result.get("final_fallback_reason"),
            "already_tried_detected": exhausted,
            "high_risk_detected": high_risk,
            "frustration_detected": frustration,
        },
    }
