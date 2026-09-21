"""
apply_policy_decisions.py — applies the two policy decisions approved by the
user:
1. "troubleshooting already exhausted" is a SIGNAL for the escalation policy
   (to be combined with severity/confidence/evidence later), not an absolute
   rule. No golden-set field changes needed for this one -- it's a policy
   note for the future escalation-engine stage, logged in decision_log.md.
2. Taxonomy stays frozen. The 4 needs_human_decision examples get explicit
   `out_of_scope` / `out_of_scope_reason` fields rather than a forced,
   misleading label. The pure-praise example additionally gets
   `excluded_from_evaluation=True` so it doesn't distort classifier metrics.
"""

import json
from pathlib import Path

RESOLUTIONS = {
    "cand_0008": {
        "intent": "Unclear / Insufficient Information",
        "out_of_scope": True,
        "out_of_scope_reason": "feature_request",
        "excluded_from_evaluation": False,
        "note": "Resolved per user policy: kept as Unclear for practical single-label evaluation, flagged out_of_scope so it's identifiable and excludable for analysis rather than silently blended into genuine 'insufficient information' cases.",
    },
    "cand_0038": {
        "intent": "Unclear / Insufficient Information",
        "out_of_scope": True,
        "out_of_scope_reason": "publisher_content_request",
        "excluded_from_evaluation": False,
        "note": "Resolved per user policy: same treatment as cand_0008.",
    },
    "cand_0183": {
        "intent": "Unclear / Insufficient Information",
        "out_of_scope": True,
        "out_of_scope_reason": "out_of_taxonomy_how_to",
        "excluded_from_evaluation": False,
        "note": "Resolved per user policy: a specific, answerable how-to question that doesn't match any of the 9 intents' definitions; kept as Unclear pragmatically, flagged out_of_scope.",
    },
    "cand_0123": {
        "out_of_scope": True,
        "out_of_scope_reason": "non_support_chatter",
        "excluded_from_evaluation": True,
        "note": "Resolved per user policy: not a support request at all (positive customer comment). Per explicit instruction, excluded from evaluation entirely rather than forced into any label, including Unclear.",
    },
}


def main():
    golden = json.loads(open("data/processed/golden_set.json").read())

    # default fields for all 189
    for g in golden:
        g.setdefault("out_of_scope", False)
        g.setdefault("out_of_scope_reason", None)
        g.setdefault("excluded_from_evaluation", False)

    by_id = {g["candidate_id"]: g for g in golden}
    for cid, res in RESOLUTIONS.items():
        item = by_id[cid]
        note = res.pop("note")
        item.update(res)
        item["_review_status"] = "llm_reviewed_pending_human_signoff"
        item["label_rationale"] = "[RESOLVED PER USER POLICY] " + note + " || " + item["label_rationale"]

    Path("data/processed/golden_set.json").write_text(json.dumps(golden, indent=2))

    n_oos = sum(1 for g in golden if g["out_of_scope"])
    n_excl = sum(1 for g in golden if g["excluded_from_evaluation"])
    print(f"out_of_scope=True: {n_oos}")
    print(f"excluded_from_evaluation=True: {n_excl}")
    print(f"needs_human_decision remaining: {sum(1 for g in golden if g['_review_status']=='needs_human_decision')}")
    print("Wrote updated data/processed/golden_set.json")


if __name__ == "__main__":
    main()
