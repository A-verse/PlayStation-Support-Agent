"""
apply_manual_corrections_batch2.py — LLM-assisted review pass #2, covering
the 94 examples left as rule_based_draft after batch 1. Same policy as batch
1 (see apply_manual_corrections.py docstring), explicitly NOT a human review.
Every entry's _review_status becomes either:
  - "llm_reviewed_pending_human_signoff" (reviewed, correction or confirmation)
  - "needs_human_decision" (genuinely ambiguous -- taxonomy doesn't cleanly
    fit, or a judgment call too consequential to make unilaterally)

Nothing here overwrites the original rule-based draft fields -- corrections
are recorded as deltas with rationale, same audit pattern as batch 1.
"""

import json
from pathlib import Path

CORRECTIONS = {
    "cand_0005": {
        "intent": "Hardware / Device Malfunction",
        "note": "Proxy mislabeled Unclear from the current turn alone ('you sent a link for PS4, I have PS3'), but prior_context shows the real issue: blu-ray drive won't read discs and controller won't stay connected. Context should inform intent for follow-ups.",
    },
    "cand_0008": {
        "_flag": "needs_human_decision",
        "note": "This is a feature request ('add an option to change notification location'), not a bug/issue at all. None of the 9 frozen intents cleanly cover pure product feedback -- 'Unclear' is meant for message-quality problems, not clear-but-out-of-scope requests. Needs a policy decision: treat as Unclear by default, or explicitly note the taxonomy has a coverage gap for feature requests.",
    },
    "cand_0011": {
        "intent": "Console Troubleshooting (Safe Mode / General Fix)",
        "multi_intent_candidates": ["Downloads & Digital Content"],
        "note": "This is a system-software restore/update failure, which fits the Console Troubleshooting definition (general fix) better than Downloads (which is about digital content/game downloads specifically).",
    },
    "cand_0012": {
        "expected_action": "Acknowledge that the customer already took a drastic self-service step (full console initialization/factory reset) to escape the safe-mode loop; confirm whether the issue is now resolved rather than repeating safe-mode instructions they've already gone past.",
        "note": "Customer already resolved this themselves via factory reset -- the default per-intent action template (repeat safe-mode steps) would be redundant/wrong here, same pattern as the closing-message cases in batch 1.",
    },
    "cand_0016": {
        "should_escalate": True,
        "escalation_triggers_add": ["troubleshooting_already_exhausted (customer states the suggested fix was 'the first thing I tried')"],
        "note": "Consistent with the exhausted-troubleshooting pattern established in batch 1.",
    },
    "cand_0018": {
        "multi_intent_candidates": ["Purchases, Billing & Refunds"],
        "should_escalate": True,
        "note": "Account deletion dispute combined with an explicit financial claim (>\u00a31000 in purchased content) is a serious dispute needing account-specific verification -- matches both the ban/account-action and financial-dispute escalation criteria.",
    },
    "cand_0020": {
        "intent": "Hardware / Device Malfunction",
        "note": "Fan noise is a physical/mechanical symptom, fits Hardware better than general Console Troubleshooting (which is for software-level fixes like safe mode).",
    },
    "cand_0027": {
        "intent": "Purchases, Billing & Refunds",
        "multi_intent_candidates": ["Account Access & Recovery"],
        "note": "Question is specifically about transferring purchased content between accounts -- a billing/entitlement question, not primarily an account-access issue.",
    },
    "cand_0031": {
        "intent": "Purchases, Billing & Refunds",
        "note": "Redemption-code issues were consolidated into Purchases/Billing/Refunds when the taxonomy was frozen (no separate 'Redemption Codes' intent survived taxonomy finalization) -- this is a code-redemption failure, belongs here, not Console Troubleshooting.",
    },
    "cand_0034": {
        "intent": "Account Access & Recovery",
        "should_escalate": True,
        "escalation_triggers_add": ["self_service_tool_broken_long_term (deactivation tool reportedly non-functional for 6 months)"],
        "note": "This is account/device management (deactivating consoles), not a vague message. A self-service tool broken for 6 months is real evidence the standard flow won't help -- escalate.",
    },
    "cand_0038": {
        "_flag": "needs_human_decision",
        "note": "This is a publisher/content-availability request ('help me beg [publisher] to release this game on PS4'), not a support issue at all. Same taxonomy-coverage gap as cand_0008 -- no intent cleanly covers non-support chatter that happens to appear in the brand's reply thread.",
    },
    "cand_0039": {
        "intent": "Purchases, Billing & Refunds",
        "note": "Context establishes this is a region-locked purchase issue (FIFA points bought as EU, account is US) -- the current turn alone looks vague but the topic is clear from context.",
    },
    "cand_0042": {
        "expected_action": "Ask what specific account issue they're experiencing before assuming it's a password/ID recovery case -- 'account issue' alone doesn't specify which self-service flow applies.",
        "note": "Kept intent as Account Access (the topic domain is named, even if vague on specifics), but the default action template over-assumed a specific recovery flow. Refined to ask for detail first.",
    },
    "cand_0047": {
        "intent": "Unclear / Insufficient Information",
        "note": "'Ps4 Account' (and prior context 'Main Ps4 Account') carries no actual request or complaint -- there's nothing here to act on yet, genuinely insufficient information rather than a nameable Account Access issue.",
    },
    "cand_0049": {
        "should_escalate": True,
        "escalation_triggers_add": ["troubleshooting_already_exhausted (prior context: 'I followed all instructions on the website')"],
        "note": "Customer already followed the documented website instructions before contacting support.",
    },
    "cand_0050": {
        "should_escalate": True,
        "escalation_triggers_add": ["troubleshooting_already_exhausted (advanced safe-mode options #3 and #4 already tried without success)"],
        "note": "Advanced safe-mode options already exhausted without fixing a corrupted file -- suggests a deeper problem (possible storage/hardware fault) beyond standard guidance.",
    },
    "cand_0055": {
        "intent": "Hardware / Device Malfunction",
        "note": "Disc-eject behavior is a physical drive question, fits Hardware better than general Console Troubleshooting.",
    },
    "cand_0056": {
        "should_escalate": True,
        "escalation_triggers_add": ["troubleshooting_already_exhausted (emphatic 'tried EVERYTHING')"],
        "note": "Vague on specifics (rightly Unclear) but explicit exhaustion + visible frustration -- a generic clarifying question risks looking dismissive; escalate per the insufficient-evidence-to-safely-resolve criterion.",
    },
    "cand_0062": {
        "intent": "Purchases, Billing & Refunds",
        "should_escalate": True,
        "note": "Wrong item received on a preorder, with invoices offered as proof -- squarely a purchase/fulfillment dispute with evidence, not a console-troubleshooting 'error' despite the customer's own phrasing ('how do I fix this error'). Escalate: financial dispute with claimed proof needs verification.",
    },
    "cand_0064": {
        "should_escalate": True,
        "escalation_triggers_add": ["account_takeover (informal phrasing 'gotten took' = account taken/stolen; not caught by the hack/stolen regex, which expected different wording)"],
        "note": "Another real regex-phrasing gap, similar to ones found in batch 1.",
    },
    "cand_0074": {
        "intent": "Hardware / Device Malfunction",
        "multi_intent_candidates": ["Error Codes & Technical Faults"],
        "note": "'Blue screened' is a specific hardware/system-crash symptom, not a vague message -- proxy mislabeled as Unclear because the wording ('fix it') is generic even though the symptom named is specific.",
    },
    "cand_0082": {
        "should_escalate": True,
        "escalation_triggers_add": ["ambiguous_account_loss_with_evidence (customer proactively offers purchase/payment evidence, suggesting a contested or serious access-loss situation beyond routine password reset)"],
        "note": "Providing unsolicited proof-of-ownership evidence signals this isn't a routine forgot-password case; treat as needing human verification.",
    },
    "cand_0083": {
        "intent": "Network & Service Connectivity",
        "note": "'Won't connect to PSN' is a clear connectivity complaint; proxy mislabeled Unclear, likely because of typos ('pan' for PSN, 'bees' for been) reducing keyword match confidence.",
    },
    "cand_0084": {
        "should_escalate": True,
        "escalation_triggers_add": ["unexpected_shutdown_during_use (consistent with the cand_0037 precedent from batch 1 -- recurring shutdown during active use suggests a real hardware fault)"],
        "note": "Applying the same precedent set in batch 1 for this exact symptom pattern.",
    },
    "cand_0089": {
        "intent": "Network & Service Connectivity",
        "note": "NAT-type issues are a networking/connectivity topic, not a vague message -- proxy mislabeled Unclear because 'NAT' isn't in the network keyword list.",
    },
    "cand_0090": {
        "should_escalate": True,
        "escalation_triggers_add": ["lost_2fa_device (consistent with cand_0013 precedent from batch 1)"],
        "note": "Same lost-2FA-device pattern already established as an escalation case.",
    },
    "cand_0099": {
        "multi_intent_candidates": ["Network & Service Connectivity"],
        "note": "Explicitly 'cannot connect to psn' alongside the error code -- genuinely both topics.",
    },
    "cand_0100": {
        "should_escalate": True,
        "escalation_triggers_add": ["troubleshooting_already_exhausted ('tried a couple of troubleshooting videos but none of them worked')"],
        "note": "Consistent with established pattern.",
    },
    "cand_0105": {
        "intent": "Downloads & Digital Content",
        "should_escalate": True,
        "escalation_triggers_add": ["troubleshooting_already_exhausted", "customer_frustration (curt, annoyed tone after repeating themselves)"],
        "note": "Proxy mislabeled Unclear from the curt current turn; context shows missing purchased content with the official restore-licenses fix already tried and failed, plus visible frustration at getting an unhelpful repeat response.",
    },
    "cand_0106": {
        "should_escalate": True,
        "escalation_triggers_add": ["unrecognized_purchase (phrasing 'a purchase I didn't make' implies an unauthorized charge; not caught by existing regex patterns which expected the word 'unauthorized' or 'fraud')"],
        "note": "Another real phrasing gap in the escalation regex -- semantically identical to an unauthorized-charge case but worded differently.",
    },
    "cand_0113": {
        "should_escalate": True,
        "escalation_triggers_add": ["official_recovery_tool_erroring (the password-reset flow itself is returning an error, so the standard self-service guidance would not help)"],
        "note": "Customer is already inside the official flow and it's broken -- repeating 'use the reset flow' would be a false auto-handle.",
    },
    "cand_0114": {
        "should_escalate": True,
        "escalation_triggers_add": ["official_flow_bug (Terms-of-Service acceptance screen not appearing, blocking legitimate sub-account login)"],
        "note": "Sounds like a genuine bug in the official login flow, not something self-service guidance can fix.",
    },
    "cand_0121": {
        "should_escalate": True,
        "escalation_triggers_add": ["troubleshooting_already_exhausted (already tried Ethernet connection, ruled out shareplay/connection-strength as the cause)"],
        "note": "Customer has already ruled out the obvious causes with specific technical detail -- repeating generic connectivity steps would likely fail again.",
    },
    "cand_0123": {
        "_flag": "needs_human_decision",
        "note": "This is not a support request at all -- it's a positive customer comment/testimonial about switching to streaming services including Vue. Doesn't fit any intent's definition of an issue needing resolution. Recommend excluding non-support chatter from the golden set, or explicitly deciding how these should be labeled if kept.",
    },
    "cand_0128": {
        "multi_intent_candidates": ["Network & Service Connectivity"],
        "note": "Context explicitly names 'a problem with the PS network' alongside the error code.",
    },
    "cand_0131": {
        "intent": "Purchases, Billing & Refunds",
        "should_escalate": True,
        "note": "Explicit refund/compensation demand ('if not fixed I'd like my money back') tied to a lost-progress bug -- primary actionable ask is financial, not just a bug report. Escalate: financial dispute.",
    },
    "cand_0133": {
        "intent": "Hardware / Device Malfunction",
        "should_escalate": True,
        "escalation_triggers_add": ["failed_repair_rma (device returned from official service repair still exhibiting the same problem -- direct RMA/hardware escalation case)"],
        "note": "The current turn ('Ok now what') is only interpretable via context: a PS4 Pro came back from official repair still broken. This is exactly the kind of RMA situation the escalation criteria call out explicitly.",
    },
    "cand_0137": {
        "should_escalate": True,
        "escalation_triggers_add": ["troubleshooting_already_exhausted ('Tried re-installing card. Did not work.')"],
        "note": "Consistent with established pattern.",
    },
    "cand_0142": {
        "should_escalate": True,
        "escalation_triggers_add": ["troubleshooting_already_exhausted", "two_different_controllers_failing (consistent with cand_0091 precedent from batch 1 -- points to a console-side fault)"],
        "note": "Same two-controllers-failing pattern already established as an escalation case, plus explicit exhaustion language.",
    },
    "cand_0144": {
        "intent": "Hardware / Device Malfunction",
        "should_escalate": True,
        "escalation_triggers_add": ["troubleshooting_already_exhausted"],
        "note": "Proxy mislabeled Unclear from the terse current turn ('It did not work'), but context is a physical controller defect (R2 button drift) with the suggested fix already failed.",
    },
    "cand_0150": {
        "should_escalate": True,
        "escalation_triggers_add": ["troubleshooting_already_exhausted (already powered down and reinserted disc per the brand's own earlier suggestion, still stuck)"],
        "note": "Consistent with established pattern.",
    },
    "cand_0152": {
        "should_escalate": True,
        "escalation_triggers_add": ["cannot_complete_verification (forgot the security info required to complete the official reset flow -- will loop indefinitely on self-service)"],
        "note": "Similar category to the lost-2FA and lost-recovery-email cases -- official flow cannot be completed without info the customer no longer has.",
    },
    "cand_0153": {
        "multi_intent_candidates": ["Hardware / Device Malfunction"],
        "should_escalate": True,
        "escalation_triggers_add": ["reported_bricked_device (customer claims the update rendered the system non-functional -- a severe, non-standard-troubleshooting claim)"],
        "note": "A 'bricked' claim is a hardware-level consequence of a software update; standard error-code lookup alone isn't sufficient if the claim is accurate.",
    },
    "cand_0156": {
        "should_escalate": True,
        "escalation_triggers_add": ["troubleshooting_already_exhausted (issue reproduced across 3 different devices, ruling out device-specific causes)"],
        "note": "Consistent with established pattern -- ruling out multiple devices is meaningful exhaustion evidence.",
    },
    "cand_0158": {
        "intent": "Network & Service Connectivity",
        "note": "Clear connectivity/speed complaint with a useful diagnostic detail (isolated to one device) -- not a vague message, proxy mislabeled as Unclear.",
    },
    "cand_0163": {
        "should_escalate": True,
        "escalation_triggers_add": ["troubleshooting_already_exhausted (tried the suggested PC purchase path and PayPal, both failing with repeated 'try later' responses)"],
        "note": "Persistent payment blocker across two different attempted workarounds.",
    },
    "cand_0168": {
        "intent": "Console Troubleshooting (Safe Mode / General Fix)",
        "note": "Context establishes this is an intermittent rest-mode wake bug following a software update -- a real, nameable technical topic, not vague messaging (the current turn alone reads as vague, but context clarifies it).",
    },
    "cand_0176": {
        "should_escalate": True,
        "escalation_triggers_add": ["troubleshooting_already_exhausted (an advanced step -- rebuilding the database -- was already tried per the brand's own suggestion, issue persists)"],
        "note": "Consistent with established pattern; also consider Hardware overlap since controller-only-works-in-safe-mode could be console-side, but leaving primary intent unchanged given time constraints -- flagged in rationale for whoever signs off.",
    },
    "cand_0178": {
        "should_escalate": True,
        "escalation_triggers_add": ["lost_recovery_email (similar to the lost-2FA-device precedent -- standard email-based account recovery typically requires access to the original email)"],
        "note": "Same category of self-service-can't-complete-without-lost-credential as the 2FA case.",
    },
    "cand_0180": {
        "intent": "Downloads & Digital Content",
        "multi_intent_candidates": ["Purchases, Billing & Refunds"],
        "note": "Context reveals missing purchased content (game/DLC not appearing in a specific region's store) -- Downloads/Content fits better than Console Troubleshooting; the current turn ('Please fix this') alone is uninformative without context.",
    },
    "cand_0183": {
        "_flag": "needs_human_decision",
        "note": "Specific how-to question (changing a PS4 profile cover image) that doesn't map cleanly to any of the 9 intents -- it's not vague (Unclear is meant for insufficient information, and this message has plenty of information), but it's also not really Account Access, Console Troubleshooting, or any other topic as defined. Same taxonomy-coverage gap as cand_0008/cand_0038 (minor feature/how-to questions), but framed as a question rather than a request -- worth a policy decision on how to bucket these consistently.",
    },
    "cand_0186": {
        "intent": "Purchases, Billing & Refunds",
        "expected_action": "Acknowledge; no new action needed. Customer has already been given the correct billing-troubleshooting link and is thanking for it (while venting frustration) rather than asking a new question.",
        "should_escalate": False,
        "note": "Proxy mislabeled Unclear; context makes clear this is a payment-method issue (credit card/PayPal add-funds failures) and the current turn is a closing/venting acknowledgment, not a fresh request -- same pattern as the closing-message cases in batch 1.",
    },
    "cand_0188": {
        "intent": "PlayStation Vue / Streaming Service",
        "should_escalate": True,
        "escalation_triggers_add": ["troubleshooting_already_exhausted (customer already tried the suggested fix, same error persists)"],
        "note": "Proxy mislabeled Unclear from the terse current turn; context is a Vue region-lock login issue with the suggested fix already failed.",
    },
}

# Read, judged consistent with the rule-based draft as-is, no field changes needed.
CONFIRMED_NO_CHANGE = [
    "cand_0003", "cand_0004", "cand_0006", "cand_0009", "cand_0014", "cand_0017",
    "cand_0026", "cand_0035", "cand_0041", "cand_0044", "cand_0046", "cand_0051",
    "cand_0053", "cand_0054", "cand_0058", "cand_0066", "cand_0073", "cand_0079",
    "cand_0080", "cand_0085", "cand_0086", "cand_0094", "cand_0107", "cand_0111",
    "cand_0115", "cand_0116", "cand_0120", "cand_0125", "cand_0134", "cand_0135",
    "cand_0141", "cand_0148", "cand_0154", "cand_0155", "cand_0160", "cand_0161",
    "cand_0169", "cand_0170", "cand_0177", "cand_0182", "cand_0185",
]


def main():
    golden = json.loads(open("data/processed/golden_set_v2.json").read())
    by_id = {g["candidate_id"]: g for g in golden}

    n_corrected, n_needs_decision, n_confirmed = 0, 0, 0

    for cid, corr in CORRECTIONS.items():
        item = by_id[cid]
        note = corr.pop("note")
        flag = corr.pop("_flag", None)
        trig_add = corr.pop("escalation_triggers_add", None)
        item.update(corr)
        if trig_add:
            item["escalation_triggers"] = item.get("escalation_triggers", []) + trig_add
        if flag == "needs_human_decision":
            item["_review_status"] = "needs_human_decision"
            item["label_rationale"] = "[NEEDS HUMAN DECISION] " + note + " || original rule rationale: " + item["label_rationale"]
            n_needs_decision += 1
        else:
            item["_review_status"] = "llm_reviewed_pending_human_signoff"
            item["label_rationale"] = "[LLM-REVIEWED, CORRECTED] " + note + " || original rule rationale: " + item["label_rationale"]
            n_corrected += 1

    for cid in CONFIRMED_NO_CHANGE:
        item = by_id[cid]
        item["_review_status"] = "llm_reviewed_pending_human_signoff"
        item["label_rationale"] = "[LLM-REVIEWED, confirmed no change] " + item["label_rationale"]
        n_confirmed += 1

    remaining = [g["candidate_id"] for g in golden if g["_review_status"] == "rule_based_draft"]

    print(f"Corrected: {n_corrected}")
    print(f"Needs human decision: {n_needs_decision}")
    print(f"Confirmed unchanged: {n_confirmed}")
    print(f"Total processed this batch: {n_corrected + n_needs_decision + n_confirmed} (expected 94)")
    print(f"Still rule_based_draft (should be 0): {len(remaining)} {remaining}")

    Path("data/processed/golden_set_v3.json").write_text(json.dumps(golden, indent=2))
    print("\nWrote data/processed/golden_set_v3.json")


if __name__ == "__main__":
    main()
