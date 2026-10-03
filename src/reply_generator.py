"""
reply_generator.py - the first production-style grounded reply generator.

Pipeline: customer message (+ prior context)
    -> intent classifier + confidence
    -> confidence-gated retrieval (intent-aware if confident, global if not)
    -> evidence-quality check
    -> grounded reply (real LLM call, with safe fallback on LLM failure,
       deterministic MOCK MODE otherwise)

The LLM prompt is strictly grounded in retrieved historical resolutions.
"""

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from retrieval import Retriever
from response_safety import (
    assess_message_quality,
    assess_response_safety,
    normalize,
    build_idf_lookup,
)
from llm_providers import get_provider


MENTION_RE = re.compile(r"@\w+")

CONFIDENCE_THRESHOLD = 0.6
WEAK_SIMILARITY_THRESHOLD = 0.30

ALREADY_TRIED_RE = re.compile(
    r"\b(already (tried|did|done)|did that|tried that|that didn'?t work|"
    r"still (doesn'?t|does not) work|tried (all|everything)|"
    r"it did(n'?t| not) work)\b",
    re.I,
)

REPLY_MODEL = os.environ.get("REPLY_GEN_MODEL")


SYSTEM_PROMPT = """You are drafting a customer-support reply for AskPlayStation on behalf of a human review process (this draft will be reviewed before sending).

You will be given:
- The customer's current message and, if relevant, prior conversation turns.
- 1-3 retrieved historical (customer message, brand reply) pairs that are the ONLY source of truth for what PlayStation support policy or troubleshooting steps are.

STRICT RULES:

1. Base your reply PRIMARILY on the retrieved historical resolutions. Do not invent PlayStation policies, refund rules, troubleshooting steps, account actions, or guarantees that are not supported by the retrieved evidence.

2. If the customer's message or prior context indicates they ALREADY TRIED a fix, do not suggest that same fix again -- acknowledge it didn't work and suggest escalation/next steps instead, only if the evidence supports a next step.

3. Never claim to have performed an account-specific action (e.g. "I've refunded you", "I've reset your account") -- you are drafting a reply, not executing an action.

4. Never mention retrieval, similarity scores, intents, classifiers, or any internal system detail to the customer. Write as a normal support agent would.

5. If the evidence is weak, conflicting, or doesn't clearly address the customer's issue, do not guess -- write a brief, genuine clarifying question instead.

6. Keep the reply concise (2-4 sentences), in a warm, professional support tone.

Output ONLY the reply text, nothing else."""


def enrich(customer_text, prior_context):
    ctx_text = " ".join(c["text"] for c in prior_context)
    return MENTION_RE.sub(" ", f"{ctx_text} {customer_text}").strip()


def detect_already_tried(customer_text, prior_context):
    """Detect whether the customer is reporting a previously attempted fix."""
    return bool(ALREADY_TRIED_RE.search(normalize(customer_text)))


class ReplyGenerator:

    def __init__(self):
        self.clf = joblib.load(
            "models/tfidf_logreg_intent_classifier.joblib"
        )

        self.retriever = Retriever()

        self.idf_lookup = build_idf_lookup(
            self.retriever.vectorizer
        )

        self.provider = get_provider()

        # None if no LLM credentials configured -> mock mode
        self.mock_mode = self.provider is None

    def classify(self, customer_text, prior_context):
        text = enrich(customer_text, prior_context)

        proba = self.clf.predict_proba([text])[0]

        idx = int(np.argmax(proba))

        return (
            self.clf.classes_[idx],
            float(proba[idx]),
        )

    def assess_evidence(
        self,
        predicted_intent,
        retrieved,
        message_quality,
    ):
        reasons = []

        if (
            not retrieved
            or retrieved[0]["cosine_sim"] < WEAK_SIMILARITY_THRESHOLD
        ):
            reasons.append("weak_similarity")

        if predicted_intent == "Unclear / Insufficient Information":
            reasons.append("unclear_intent")

        top3_intents = [
            r["predicted_intent"]
            for r in retrieved[:3]
        ]

        if (
            top3_intents
            and max(Counter(top3_intents).values()) < 2
        ):
            reasons.append("conflicting_intents")

        # Message-quality guard
        if message_quality["override_to_low_confidence"]:
            reasons.append(
                "message_quality_guard:"
                + "+".join(message_quality["reasons"])
            )

        return len(reasons) > 0, reasons

    def build_prompt(
        self,
        customer_text,
        prior_context,
        retrieved,
        already_tried,
    ):
        ctx_block = ""

        if prior_context:
            ctx_lines = "\n".join(
                f"  [{c['role']}] {c['text']}"
                for c in prior_context
            )

            ctx_block = (
                f"Prior conversation:\n"
                f"{ctx_lines}\n\n"
            )

        evidence_block = ""

        for i, r in enumerate(retrieved[:3], 1):
            evidence_block += (
                f"Historical example {i} "
                f"(similarity {r['cosine_sim']:.2f}):\n"
                f"  Customer: {r['customer_text']}\n"
                f"  Brand reply: {r['brand_text']}\n\n"
            )

        already_tried_note = (
            "\nNOTE: The customer's message indicates they ALREADY "
            "TRIED a suggested fix and it did not work. "
            "Do not repeat that fix.\n"
            if already_tried
            else ""
        )

        return (
            f"{ctx_block}"
            f"Customer's current message: "
            f"{customer_text}\n\n"
            f"Retrieved historical resolutions:\n"
            f"{evidence_block}"
            f"{already_tried_note}\n"
            f"Draft a grounded reply."
        )

    def safe_fallback_reply(self, already_tried):
        """
        Single source of truth for safe fallback text.
        Used when evidence is weak, safety checks fail,
        or the external LLM is temporarily unavailable.
        """

        if already_tried:
            return (
                "Sorry that didn't resolve it. Could you share a bit "
                "more detail -- any error code you're seeing, or exactly "
                "what happens when you try again? That'll help us point "
                "you to the right next step."
            )

        return (
            "Thanks for reaching out. Could you share a few more "
            "details -- what device you're using and exactly what's "
            "happening -- so we can help you more precisely?"
        )

    def generate_mock_reply(
        self,
        customer_text,
        prior_context,
        predicted_intent,
        retrieved,
        already_tried,
    ):
        """
        Deterministic template-based best-effort reply.
        """

        if not retrieved:
            return self.safe_fallback_reply(already_tried)

        top = retrieved[0]

        if already_tried:
            return (
                "Sorry to hear that didn't work. Since the usual "
                "first step didn't help, please follow us and send "
                "a Direct Message with more details so we can look "
                "into this further."
            )

        return top["brand_text"].strip()

    def generate_llm_reply(
        self,
        customer_text,
        prior_context,
        retrieved,
        already_tried,
    ):
        prompt = self.build_prompt(
            customer_text,
            prior_context,
            retrieved,
            already_tried,
        )

        try:
            reply = self.provider.complete(
                system=SYSTEM_PROMPT,
                user=prompt,
                max_tokens=300,
            )

            return reply, None

        except Exception as exc:
            # Do not crash the Streamlit app if Gemini or another
            # configured LLM provider temporarily fails.
            return None, {
                "type": type(exc).__name__,
                "message": str(exc),
            }

    def generate(
        self,
        customer_text,
        prior_context=None,
        k=3,
    ):
        prior_context = prior_context or []

        predicted_intent, confidence = self.classify(
            customer_text,
            prior_context,
        )

        # Layer 1: message-quality guard
        message_quality = assess_message_quality(
            customer_text,
            prior_context,
        )

        effective_low_confidence = (
            confidence < CONFIDENCE_THRESHOLD
            or message_quality["override_to_low_confidence"]
        )

        retrieval_mode = (
            "global"
            if effective_low_confidence
            else "intent_aware"
        )

        retrieved = self.retriever.retrieve(
            customer_text,
            prior_context,
            k=k,
            mode=retrieval_mode,
            query_intent=predicted_intent,
        )

        evidence_failure, evidence_failure_reasons = (
            self.assess_evidence(
                predicted_intent,
                retrieved,
                message_quality,
            )
        )

        already_tried = detect_already_tried(
            customer_text,
            prior_context,
        )

        # Track LLM errors explicitly.
        # This is important because a fallback caused by Gemini
        # being unavailable must not look like an LLM-generated reply.
        llm_error = None

        if self.mock_mode:

            candidate_reply = self.generate_mock_reply(
                customer_text,
                prior_context,
                predicted_intent,
                retrieved,
                already_tried,
            )

            generation_method = "mock_template"

        else:

            candidate_reply, llm_error = self.generate_llm_reply(
                customer_text,
                prior_context,
                retrieved,
                already_tried,
            )

            if llm_error:

                # Gemini/API failure:
                # use the deterministic safe fallback instead of
                # crashing the application.
                candidate_reply = self.safe_fallback_reply(
                    already_tried
                )

                generation_method = (
                    f"llm_error_fallback:{self.provider.name}"
                )

            else:

                generation_method = (
                    f"llm:{self.provider.name}:"
                    f"{self.provider.model}"
                )

        # Layer 2: response-safety check
        top_retrieved = (
            retrieved[0]
            if retrieved
            else None
        )

        safety = assess_response_safety(
            customer_text,
            prior_context,
            candidate_reply,
            top_retrieved,
            already_tried,
            idf_lookup=self.idf_lookup,
        )

        # Centralized fallback decision
        final_fallback_reason = None

        if safety["response_safety_failure"]:

            final_reply = self.safe_fallback_reply(
                already_tried
            )

            final_fallback_reason = (
                "response_safety_failure:"
                + "+".join(
                    safety["failed_checks"]
                )
            )

        elif evidence_failure:

            final_reply = self.safe_fallback_reply(
                already_tried
            )

            final_fallback_reason = (
                "evidence_failure:"
                + "+".join(
                    evidence_failure_reasons
                )
            )

        elif llm_error:

            final_reply = self.safe_fallback_reply(
                already_tried
            )

            final_fallback_reason = (
                "llm_error:"
                + llm_error["type"]
            )

        else:

            final_reply = candidate_reply

        return {
            "customer_text": customer_text,

            "prior_context": prior_context,

            "predicted_intent": predicted_intent,

            "classifier_confidence": confidence,

            "message_quality": message_quality,

            "retrieval_mode": retrieval_mode,

            "retrieved_evidence": [
                {
                    "pair_id": r["pair_id"],
                    "historical_customer_text": r["customer_text"],
                    "historical_brand_reply": r["brand_text"],
                    "similarity_score": r["cosine_sim"],
                    "predicted_intent": r["predicted_intent"],
                }
                for r in retrieved
            ],

            "evidence_failure": evidence_failure,

            "evidence_failure_reasons": (
                evidence_failure_reasons
            ),

            "already_tried_detected": already_tried,

            "candidate_reply": candidate_reply,

            "polarity_check": (
                safety["polarity_check"]
            ),

            "responsiveness_check": (
                safety["responsiveness_check"]
            ),

            "context_consistency_check": (
                safety["context_consistency_check"]
            ),

            "response_safety_failure": (
                safety["response_safety_failure"]
            ),

            "final_fallback_reason": (
                final_fallback_reason
            ),

            "generated_reply": final_reply,

            "generation_metadata": {
                "method": generation_method,

                "mock_mode": self.mock_mode,

                "llm_provider": (
                    self.provider.name
                    if self.provider
                    else None
                ),

                "llm_error": llm_error,

                "confidence_threshold": (
                    CONFIDENCE_THRESHOLD
                ),

                "weak_similarity_threshold": (
                    WEAK_SIMILARITY_THRESHOLD
                ),
            },
        }


if __name__ == "__main__":

    gen = ReplyGenerator()

    print(
        f"Mock mode: {gen.mock_mode}\n"
    )

    demo = gen.generate(
        "my ps4 controller won't charge and turns off randomly"
    )

    print(
        json.dumps(
            demo,
            indent=2
        )[:2000]
    )