"""
test_escalation_engine.py — covers the 5 scenarios explicitly required:
normal resolvable requests, ambiguous messages, repeated failed
troubleshooting, frustrated-but-resolvable customers, and safety-check
fallbacks. Also covers the "frustration alone must not trigger escalation"
policy directly, and the high-risk-intent override.

Run with: pytest tests/test_escalation_engine.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from escalation_engine import decide, detect_frustration, detect_high_risk


def make_result(**overrides):
    """Builds a minimal ReplyGenerator.generate()-shaped dict with sane
    defaults, so each test only needs to override what it's testing."""
    base = {
        "customer_text": "how do I fix this",
        "prior_context": [],
        "predicted_intent": "Console Troubleshooting (Safe Mode / General Fix)",
        "classifier_confidence": 0.85,
        "evidence_failure": False,
        "evidence_failure_reasons": [],
        "response_safety_failure": False,
        "final_fallback_reason": None,
        "already_tried_detected": False,
    }
    base.update(overrides)
    return base


class TestNormalResolvableRequest:
    def test_high_confidence_good_evidence_auto_handles(self):
        result = make_result(
            customer_text="how do I get out of safe mode?",
            classifier_confidence=0.92,
        )
        decision = decide(result)
        assert decision["action"] == "auto_handle"
        assert decision["sub_mode"] == "resolve"
        assert decision["escalation_score"] == 0
        assert decision["reason_codes"] == []


class TestAmbiguousMessage:
    def test_unclear_intent_alone_does_not_escalate(self):
        """A genuinely vague message (Unclear intent, evidence_failure only
        because of that) should get a safe clarifying question, not a human
        escalation -- per the taxonomy-stage design decision that Unclear
        is a routing state, not an automatic escalation trigger."""
        result = make_result(
            customer_text="i need help",
            predicted_intent="Unclear / Insufficient Information",
            classifier_confidence=0.55,
            evidence_failure=True,
            evidence_failure_reasons=["unclear_intent"],
        )
        decision = decide(result)
        assert decision["action"] == "auto_handle"
        assert decision["sub_mode"] == "clarify"
        # low confidence (0.55 < 0.6) is still logged as a signal even
        # though it alone isn't enough to cross the threshold
        assert "LOW_CLASSIFIER_CONFIDENCE" in decision["reason_codes"]
        assert "INSUFFICIENT_EVIDENCE_ON_REAL_INTENT" not in decision["reason_codes"]

    def test_unclear_plus_low_confidence_plus_safety_fallback_escalates(self):
        """Multiple weak signals compounding SHOULD cross the threshold,
        even though no single one of them would alone."""
        result = make_result(
            customer_text="wtf??",
            predicted_intent="Unclear / Insufficient Information",
            classifier_confidence=0.4,
            evidence_failure=True,
            evidence_failure_reasons=["unclear_intent", "weak_similarity"],
            response_safety_failure=True,
        )
        decision = decide(result)
        assert decision["action"] == "escalate"
        assert decision["escalation_score"] >= 2


class TestRepeatedFailedTroubleshooting:
    def test_already_tried_alone_does_not_escalate(self):
        """Exhausted troubleshooting is a SIGNAL, not an absolute rule --
        per the approved policy, it must combine with other signals."""
        result = make_result(
            customer_text="already tried that, still won't connect",
            predicted_intent="Hardware / Device Malfunction",
            classifier_confidence=0.9,
            already_tried_detected=True,
        )
        decision = decide(result)
        assert decision["action"] == "auto_handle"
        assert "EXHAUSTED_TROUBLESHOOTING_SIGNAL" in decision["reason_codes"]

    def test_already_tried_plus_weak_evidence_escalates(self):
        result = make_result(
            customer_text="tried everything, still doesn't work",
            predicted_intent="Network & Service Connectivity",
            classifier_confidence=0.5,
            evidence_failure=True,
            evidence_failure_reasons=["weak_similarity"],
            already_tried_detected=True,
        )
        decision = decide(result)
        assert decision["action"] == "escalate"
        assert "EXHAUSTED_TROUBLESHOOTING_SIGNAL" in decision["reason_codes"]
        assert "LOW_CLASSIFIER_CONFIDENCE" in decision["reason_codes"]


class TestFrustratedButResolvable:
    def test_frustration_alone_never_escalates(self):
        result = make_result(
            customer_text="this is ridiculous, my controller won't charge!!",
            predicted_intent="Hardware / Device Malfunction",
            classifier_confidence=0.91,
        )
        decision = decide(result)
        assert decision["action"] == "auto_handle"
        assert "FRUSTRATION_DETECTED" in decision["reason_codes"]
        # frustration contributes 0 points -- verify the score reflects that
        assert decision["escalation_score"] == 0

    def test_frustration_detector_fires_on_expected_patterns(self):
        assert detect_frustration("this is ridiculous!!")
        assert detect_frustration("worst support ever")
        assert not detect_frustration("how do I reset my password")


class TestSafetyCheckFallback:
    def test_safety_fallback_alone_auto_handles_with_clarification(self):
        result = make_result(
            response_safety_failure=True,
            final_fallback_reason="response_safety_failure:polarity",
        )
        decision = decide(result)
        assert decision["action"] == "auto_handle"
        assert decision["sub_mode"] == "clarify"
        assert "SAFETY_CHECK_FALLBACK" in decision["reason_codes"]

    def test_safety_fallback_plus_high_risk_escalates(self):
        result = make_result(
            customer_text="my account was hacked",
            predicted_intent="Account Access & Recovery",
            response_safety_failure=True,
        )
        decision = decide(result)
        assert decision["action"] == "escalate"
        assert "HIGH_RISK_INTENT_SIGNAL" in decision["reason_codes"]


class TestHighRiskOverride:
    def test_high_risk_escalates_even_with_high_confidence_and_good_evidence(self):
        result = make_result(
            customer_text="my account was hacked and items were stolen",
            predicted_intent="Account Access & Recovery",
            classifier_confidence=0.95,
        )
        decision = decide(result)
        assert decision["action"] == "escalate"
        assert "HIGH_RISK_INTENT_SIGNAL" in decision["reason_codes"]

    def test_high_risk_detector_examples(self):
        assert detect_high_risk("my account got hacked", [])
        assert detect_high_risk("I want a refund, this is a dispute", [])
        assert detect_high_risk("my ps4 is bricked", [])
        assert not detect_high_risk("how do I get out of safe mode", [])


class TestDecisionAuditTrail:
    def test_decision_includes_full_inputs_for_audit(self):
        result = make_result()
        decision = decide(result)
        assert "decision_inputs" in decision
        assert "signal_breakdown" in decision
        assert decision["decision_inputs"]["customer_text"] == result["customer_text"]
        assert set(decision["signal_breakdown"].keys()) == {
            "HIGH_RISK_INTENT_SIGNAL", "INSUFFICIENT_EVIDENCE_ON_REAL_INTENT",
            "LOW_CLASSIFIER_CONFIDENCE", "SAFETY_CHECK_FALLBACK",
            "EXHAUSTED_TROUBLESHOOTING_SIGNAL", "FRUSTRATION_DETECTED",
        }
