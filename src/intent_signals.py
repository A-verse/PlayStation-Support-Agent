"""
intent_signals.py — regex/keyword signal detection used ONLY to stratify
sampling for the golden set. This is NOT the final classifier and NOT ground
truth. It exists so we can (a) pull enough candidates from rare intents like
Hardware/Vue/Downloads, and (b) flag messages that hit multiple intents'
signals as likely multi-intent candidates for a human to review.

Patterns are derived directly from the EDA cluster top-terms and the earlier
targeted hardware-keyword search — not invented from generic assumptions.
"""

import re

# Order matters only for picking a single "proxy_intent" (stratification
# convenience); `matched_intents` below returns ALL matches regardless of order.
INTENT_PATTERNS = {
    "PlayStation Vue / Streaming Service": [r"\bvue\b", r"\bplaystation vue\b"],
    "Hardware / Device Malfunction": [
        r"overheat", r"won.?t turn on", r"turns? off\b", r"shutting down", r"shuts? off",
        r"controller.{0,15}(charg|sync|pair|connect)", r"disc.{0,15}(stuck|eject|scratch|won.?t|drive|read)",
        r"blue light", r"beeps?\b", r"fan (noise|loud)", r"\bhdmi\b", r"console.{0,15}(broken|dead|crack)",
    ],
    "Error Codes & Technical Faults": [
        r"error code", r"\bce-\d", r"\bws-\d", r"\bsu-\d", r"\bnp-\d", r"\bnw-\d", r"\bne-\d", r"\b[a-z]{2}-\d{4,6}\b",
    ],
    "Network & Service Connectivity": [
        r"\bpsn\b.{0,15}\bdown\b", r"network.{0,10}down", r"can.?t connect", r"connection (failed|error)",
        r"\bserver(s)? (down|issue)", r"\bmaintenance\b", r"live chat", r"power cycle",
    ],
    "Purchases, Billing & Refunds": [
        r"\brefund", r"credit card", r"debit card", r"\bpaypal\b", r"\bps plus\b", r"fifa points",
        r"\bcharge(d)?\b", r"\bpayment\b", r"\bpurchase(d)?\b", r"\bbought\b",
    ],
    "Account Access & Recovery": [
        r"\bpassword\b", r"\blog(ged)?.?in\b", r"\bsign.?in\b", r"\baccount\b", r"\bbanned\b",
        r"\bsuspend(ed)?\b", r"\bhacked\b", r"psn id", r"\blocked out\b",
    ],
    "Downloads & Digital Content": [
        r"\bdownload(ing)?\b", r"\binstall(ing)?\b", r"\blibrary\b", r"stuck at \d", r"won.?t install",
    ],
    "Console Troubleshooting (Safe Mode / General Fix)": [
        r"safe mode", r"restore default", r"factory reset", r"how (do|can) i fix", r"\bfix this\b",
    ],
}


def matched_intents(text: str) -> list:
    """All intents whose signal patterns fire on this text (order = taxonomy order above)."""
    hits = []
    for intent, patterns in INTENT_PATTERNS.items():
        if any(re.search(p, text, re.I) for p in patterns):
            hits.append(intent)
    return hits


def proxy_intent(text: str) -> str:
    """Single best-guess bucket for stratified sampling only. Priority order
    reflects specificity: a Vue mention or a hardware symptom is a stronger,
    less-overlapping signal than a generic 'account' mention, so those go
    first; Account Access is deliberately late since 'account' is a common
    word that co-occurs with many other intents."""
    priority = [
        "PlayStation Vue / Streaming Service",
        "Hardware / Device Malfunction",
        "Error Codes & Technical Faults",
        "Network & Service Connectivity",
        "Downloads & Digital Content",
        "Purchases, Billing & Refunds",
        "Account Access & Recovery",
        "Console Troubleshooting (Safe Mode / General Fix)",
    ]
    hits = set(matched_intents(text))
    for intent in priority:
        if intent in hits:
            return intent
    return "Unclear / Insufficient Information"
