"""
response_safety.py — safeguards added after the smoke test exposed real
failures. Two distinct layers, run at two different points in the pipeline:

1. MESSAGE-QUALITY GUARD (pre-generation): looks at the customer message
   itself, independent of classifier confidence, to catch cases like
   cand_0047 ("Ps4 Account" classified Account Access at 92% confidence but
   genuinely too short/vague to trust). This does NOT retrain or touch the
   classifier -- it's a separate policy layer that can override the
   confidence-gating decision.

2. RESPONSE-SAFETY CHECK (post-generation): looks at the CANDIDATE REPLY
   against the customer's message, prior context, and retrieved evidence,
   to catch polarity mismatches (cand_0083's "Great to hear!" to a
   complaint), non-responsive answers (cand_0035), and premature resolution
   claims (cand_0012). If this check fails, the candidate reply is
   discarded and a safe fallback is used instead.

Both layers are deliberately simple keyword/heuristic checks, not an LLM
reranker or classifier -- per instruction, only reach for something heavier
if these demonstrably fail to catch the target failure modes. Every check
result is returned, not just the final pass/fail, so failures stay
observable rather than hidden inside a single boolean.
"""

import re

# ---------------------------------------------------------------------------
# Text normalization -- applied before every regex check below. Real customer
# text (and this corpus) frequently uses curly/smart quotes ('won't' as
# 'won\u2019t'), which plain ASCII-apostrophe regexes silently fail to match.
# This was found as a genuine bug during the safeguard smoke test (several
# "no_actionable_request" and word-boundary false positives traced back to
# this), not a hypothetical -- fixing it here rather than patching every
# individual regex with awkward character classes.
# ---------------------------------------------------------------------------

_QUOTE_NORMALIZE_TABLE = str.maketrans({
    "\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
})


def normalize(text):
    return text.translate(_QUOTE_NORMALIZE_TABLE)

# ---------------------------------------------------------------------------
# Word/phrase lists -- kept small and specific, not exhaustive NLU
# ---------------------------------------------------------------------------

PROBLEM_INDICATOR_RE = re.compile(
    r"\b(error|broken|won'?t|can'?t|cannot|doesn'?t|does not|didn'?t|did not|issue|problem|"
    r"blocked|down|stuck|failed|fail|wrong|stopped|stop(s|ped)?|froze|frozen|freeze|crash(ed|ing)?|"
    r"help|please|need|not working|isn'?t working|bricked|banned|hacked|stolen|refund|lost|dead|"
    r"blue light|still (dead|down|broken|stuck))\b", re.I
)
QUESTION_RE = re.compile(r"\?|^\s*(how|why|when|what|where|is|are|does|do|can|could|would|will)\b", re.I)

CELEBRATORY_OR_CLOSING_RE = re.compile(
    r"\b(great to hear|glad to hear|awesome|perfect|glad (we|it)|you'?re welcome|no problem|"
    r"anytime|happy to help[!.]?$|great news|congrat)\b", re.I
)
CLOSING_ONLY_RE = re.compile(
    r"^(thanks?( you)?[!.,]?|thank you[!.,]?|no worries[!.,]?|you'?re welcome[!.,]?|anytime[!.,]?|"
    r"glad (we|it) could help[!.,]?)\s*$", re.I
)
RESOLUTION_CLAIM_RE = re.compile(
    r"\b(glad (that|it'?s)? ?(worked|resolved|fixed)|that'?s (fixed|resolved|sorted)|problem solved|"
    r"all (set|sorted|fixed) then)\b", re.I
)
CUSTOMER_CONFIRMS_RESOLUTION_RE = re.compile(
    r"\b(fixed|solved|resolved|worked|works now|working now|thanks?,? (it|that) (works|worked|helped)|"
    r"all good now|sorted now)\b", re.I
)
SUGGESTS_NEW_FIX_RE = re.compile(
    r"\b(try|please try|follow these steps|restart|reset|power cycle|reinstall|re-?install|"
    r"check the (next|following) link)\b", re.I
)
ACKNOWLEDGES_FAILURE_RE = re.compile(
    r"\b(sorry (to hear|that)|since that didn'?t|apologi[sz]e|we understand|let'?s look into|"
    r"send (us )?a (direct message|dm)|escalat)\b", re.I
)
BOILERPLATE_RE = re.compile(
    r"^(please (follow|check)|check your dm|we have sent you a|please follow us|"
    r"for (further|more) (assistance|information)|glad to help\.?$)", re.I
)
CLOSING_MESSAGE_RE = re.compile(
    r"^\s*(thanks?( you)?[!.,]?( so much)?|ok(ay)?[!.,]?|cool[!.,]?|got it[!.,]?|appreciate it[!.,]?|"
    r"sounds good[!.,]?)\s*$", re.I
)

STOPWORDS = set("""a an the is are was were be been being to of in on at for with and or but if this that
these those i my me you your it its we our us they them he she his her @ ps4 ps3 playstation askplaystation""".split())


def _stem(word):
    """Lightweight, dependency-free suffix stripping -- not a real stemmer,
    just enough to collapse the common inflections that caused false
    'topic mismatch' flags (install/installing, try/tried, connect/
    connecting). Intentionally conservative: only strips when the result
    stays a reasonable word length, to avoid over-stripping short words."""
    if len(word) >= 5 and word.endswith("ied"):
        return word[:-3] + "y"          # tried -> try, applied -> apply
    if len(word) >= 6 and word.endswith("ing"):
        return word[:-3]                # installing -> install, connecting -> connect, trying -> try
    if len(word) > 5 and word.endswith("ed"):
        return word[:-2]                # connected -> connect, downloaded -> download
    if len(word) > 5 and word.endswith("es"):
        return word[:-2]                # matches -> match
    if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]                # controllers -> controller
    return word


def _content_words(text):
    text = normalize(text)
    words = re.findall(r"[a-zA-Z]{4,}", text.lower())
    return set(_stem(w) for w in words if w not in STOPWORDS)


BACKWARD_REFERENCE_RE = re.compile(
    r"\b(already (tried|did|done|said|mentioned)|did that|tried that|that didn'?t work|"
    r"still (doesn'?t|does not) work|tried (all|everything)|it did(n'?t| not) work|"
    r"original (tweet|message|post)|read (the|my) (entire|whole)|as (i|I) (said|mentioned))\b", re.I
)


def _context_and_current_words(current_text, prior_context):
    """Content words for the query side of the overlap check.

    Only supplements with prior_context when the current turn contains a
    BACKWARD REFERENCE to something already said/tried ("I did that",
    "read my original tweet", "that didn't work") -- i.e. a genuine
    context-dependent follow-up whose real topic lives in earlier turns.

    This is deliberately NOT gated on message length: a message can be
    short or long and still be self-sufficient (e.g. cand_0035's 18-word
    question stands on its own), and folding in context indiscriminately
    caused a real regression found during testing -- cand_0035's own prior
    context happens to mention "safe mode" (a near-universal phrase across
    Console Troubleshooting conversations) purely because of what the BRAND
    suggested earlier, which spuriously overlapped with the retrieved
    doc's unrelated context and masked a genuine topic mismatch. Gating on
    an explicit backward-reference phrase, rather than length or
    followup-ness alone, targets the actual thing that makes a follow-up
    context-dependent: the customer pointing back at something already
    said, not just the presence of any prior turn."""
    own_words = _content_words(current_text)
    if not BACKWARD_REFERENCE_RE.search(normalize(current_text)):
        return own_words
    ctx_text = " ".join(c["text"] for c in (prior_context or []))
    return own_words | _content_words(ctx_text)


# ---------------------------------------------------------------------------
# 1. Message-quality guard (pre-generation)
# ---------------------------------------------------------------------------

def assess_message_quality(customer_text, prior_context, very_short_words=4, terse_followup_words=6):
    customer_text = normalize(customer_text)
    reasons = []
    word_count = len(customer_text.split())

    is_very_short = word_count <= very_short_words
    if is_very_short:
        reasons.append("very_short_message")

    has_actionable_signal = bool(PROBLEM_INDICATOR_RE.search(customer_text) or QUESTION_RE.search(customer_text))
    if not has_actionable_signal:
        reasons.append("no_actionable_request")

    is_followup = len(prior_context) > 0
    is_terse_followup = is_followup and word_count <= terse_followup_words
    if is_terse_followup:
        reasons.append("context_dependent_terse_followup")

    is_closing = bool(CLOSING_MESSAGE_RE.match(customer_text.strip()))
    if is_closing:
        reasons.append("closing_message")

    # Any of these override trust in the classifier's confidence, regardless
    # of how high that confidence was (this is the cand_0047 fix).
    override_to_low_confidence = is_very_short or (not has_actionable_signal) or is_terse_followup or is_closing

    return {
        "word_count": word_count,
        "is_very_short": is_very_short,
        "has_actionable_signal": has_actionable_signal,
        "is_context_dependent_terse_followup": is_terse_followup,
        "is_closing_message": is_closing,
        "override_to_low_confidence": override_to_low_confidence,
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# 2. Response-safety check (post-generation)
# ---------------------------------------------------------------------------

def check_polarity(customer_text, reply):
    """Detects: complaint -> celebratory reply; question -> closing-only
    reply; reply claims resolution the customer never confirmed."""
    customer_text = normalize(customer_text)
    reply = normalize(reply)
    reasons = []
    customer_is_problem_report = bool(PROBLEM_INDICATOR_RE.search(customer_text))
    customer_is_question = bool(QUESTION_RE.search(customer_text))
    customer_confirms_resolution = bool(CUSTOMER_CONFIRMS_RESOLUTION_RE.search(customer_text))

    reply_is_celebratory = bool(CELEBRATORY_OR_CLOSING_RE.search(reply))
    reply_is_closing_only = bool(CLOSING_ONLY_RE.match(reply.strip()))
    reply_claims_resolution = bool(RESOLUTION_CLAIM_RE.search(reply))

    if customer_is_problem_report and reply_is_celebratory and not customer_confirms_resolution:
        reasons.append("complaint_met_with_celebratory_reply")
    if customer_is_question and reply_is_closing_only:
        reasons.append("question_answered_with_closing_only")
    if (reply_is_celebratory or reply_claims_resolution) and not customer_confirms_resolution:
        reasons.append("claims_unconfirmed_resolution")

    return {"passed": len(reasons) == 0, "reasons": reasons}


def check_responsiveness(customer_text, prior_context, reply, top_retrieved, idf_lookup=None, idf_overlap_threshold=5.0):
    """Detects: reply is pure boilerplate/link-sharing with no specific
    content; reply's matched historical query shares no real topic overlap
    with the current query (the cand_0035 case: retrieval found something
    lexically close-ish but not actually about the same problem).

    Both sides of the overlap check use FULL context (see
    `_context_and_current_words` for the query-side gating rule and
    `top_retrieved["prior_context"]` for the doc side).

    IDF-WEIGHTED OVERLAP, not a flat set intersection: a flat check treats
    "safe"/"mode" (near-universal across Console Troubleshooting
    conversations, since it's the standard first-line fix) the same as a
    rare, specific term like "spotify" or "horizon zero dawn". That caused
    a real false negative during testing: cand_0035's mismatched retrieved
    doc shares only generic boilerplate words with the query, which a flat
    check counted as "overlap" and let through. Requiring at least one
    shared word with IDF >= idf_overlap_threshold means only a genuinely
    specific/rare shared term counts as real topical overlap -- common
    words shared between almost any two conversations in the same intent
    don't. idf_lookup is the retriever's own fitted TF-IDF vectorizer's
    vocabulary/idf_ (reused, not a new model), so this stays "simple TF-IDF"
    rather than adding new NLP infrastructure. If idf_lookup isn't provided,
    falls back to a flat set intersection (still usable standalone/in tests)."""
    customer_text = normalize(customer_text)
    reply = normalize(reply)
    reasons = []
    is_boilerplate = bool(BOILERPLATE_RE.match(reply.strip()))
    if is_boilerplate:
        reasons.append("boilerplate_low_specificity")

    if top_retrieved:
        query_words = _context_and_current_words(customer_text, prior_context)
        historical_ctx_text = " ".join(c["text"] for c in top_retrieved.get("prior_context", []))
        historical_full_text = f"{historical_ctx_text} {top_retrieved['customer_text']}"
        historical_query_words = _content_words(historical_full_text)
        overlap = query_words & historical_query_words

        if idf_lookup is not None:
            max_overlap_idf = max((idf_lookup.get(w, 0.0) for w in overlap), default=0.0)
            meaningful_overlap = max_overlap_idf >= idf_overlap_threshold
        else:
            meaningful_overlap = bool(overlap)

        if query_words and not meaningful_overlap:
            reasons.append("topic_mismatch_with_retrieved_precedent")

    # Boilerplate alone isn't necessarily wrong (a lot of correct historical
    # replies really are "check your DMs"); it's only a problem in
    # combination with topic mismatch, which is what the "failed" check below
    # actually gates on -- boilerplate alone is recorded but not disqualifying.
    disqualifying = "topic_mismatch_with_retrieved_precedent" in reasons
    return {"passed": not disqualifying, "reasons": reasons}


def check_context_consistency(reply, already_tried):
    """Detects: customer already reported a fix failed, but the candidate
    reply suggests trying a(nother) fix without acknowledging the failure."""
    reply = normalize(reply)
    reasons = []
    if already_tried:
        suggests_new_fix = bool(SUGGESTS_NEW_FIX_RE.search(reply))
        acknowledges_failure = bool(ACKNOWLEDGES_FAILURE_RE.search(reply))
        if suggests_new_fix and not acknowledges_failure:
            reasons.append("repeats_troubleshooting_after_failure")
    return {"passed": len(reasons) == 0, "reasons": reasons}


def build_idf_lookup(vectorizer):
    """Builds a stem -> min(idf) lookup from an already-fitted
    TfidfVectorizer's vocabulary, for use by check_responsiveness. Reuses
    the retriever's own fitted vectorizer (unigrams only; bigrams are
    skipped since content words are single stemmed tokens) -- no new model,
    just a preprocessing pass over vocabulary already computed."""
    lookup = {}
    for term, idx in vectorizer.vocabulary_.items():
        if " " in term:
            continue  # bigram, not applicable to single-token overlap check
        stem = _stem(term)
        idf = vectorizer.idf_[idx]
        if stem not in lookup or idf < lookup[stem]:
            lookup[stem] = idf
    return lookup


def assess_response_safety(customer_text, prior_context, reply, top_retrieved, already_tried, idf_lookup=None):
    polarity = check_polarity(customer_text, reply)
    responsiveness = check_responsiveness(customer_text, prior_context, reply, top_retrieved, idf_lookup=idf_lookup)
    context_consistency = check_context_consistency(reply, already_tried)

    failed_checks = [name for name, result in
                      [("polarity", polarity), ("responsiveness", responsiveness), ("context_consistency", context_consistency)]
                      if not result["passed"]]

    return {
        "polarity_check": polarity,
        "responsiveness_check": responsiveness,
        "context_consistency_check": context_consistency,
        "response_safety_failure": len(failed_checks) > 0,
        "failed_checks": failed_checks,
    }
