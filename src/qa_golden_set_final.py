import json
from collections import Counter
from pathlib import Path

golden = json.loads(open("data/processed/golden_set_v3.json").read())
n = len(golden)

print(f"=== A. Review completeness ===")
status_counts = Counter(g["_review_status"] for g in golden)
for s, c in status_counts.items():
    print(f"  {s:<40} {c}")
reviewed = sum(1 for g in golden if g["_review_status"] in ("llm_reviewed_pending_human_signoff", "needs_human_decision"))
print(f"  Reviewed (either status): {reviewed}/{n}")

print(f"\n=== B. Final intent distribution ===")
intent_counts = Counter(g["intent"] for g in golden)
for intent, c in sorted(intent_counts.items(), key=lambda x: -x[1]):
    print(f"  {intent:<55} {c:>4}  ({c/n:.1%})")

print(f"\n=== C. Multi-intent count ===")
multi = sum(1 for g in golden if g["multi_intent_candidates"])
print(f"  {multi} ({multi/n:.1%})")

print(f"\n=== D. Follow-up/context-dependent count ===")
followup = sum(1 for g in golden if g["is_followup"])
print(f"  {followup} ({followup/n:.1%})")

print(f"\n=== E. Escalation count ===")
esc = sum(1 for g in golden if g["should_escalate"])
print(f"  {esc} ({esc/n:.1%})")

print(f"\n=== F. Duplicate / near-duplicate check ===")
seen = {}
dupes = 0
for g in golden:
    key = g["customer_text"].lower().strip()
    if key in seen:
        dupes += 1
        print(f"  DUPLICATE: {g['candidate_id']} == {seen[key]}")
    seen[key] = g["candidate_id"]
print(f"  Exact duplicates: {dupes}")

print(f"\n=== G. Schema validation ===")
required_fields = ["candidate_id", "pair_id", "customer_text", "prior_context", "intent",
                    "multi_intent_candidates", "should_escalate", "expected_action",
                    "label_rationale", "_review_status"]
missing = 0
valid_intents = {
    "Account Access & Recovery", "Purchases, Billing & Refunds", "Error Codes & Technical Faults",
    "Network & Service Connectivity", "Console Troubleshooting (Safe Mode / General Fix)",
    "Downloads & Digital Content", "PlayStation Vue / Streaming Service",
    "Hardware / Device Malfunction", "Unclear / Insufficient Information"
}
bad_intents = 0
for g in golden:
    for f in required_fields:
        if f not in g:
            print(f"  MISSING FIELD {f} in {g['candidate_id']}")
            missing += 1
    if g["intent"] not in valid_intents:
        print(f"  INVALID INTENT '{g['intent']}' in {g['candidate_id']}")
        bad_intents += 1
    for mi in g["multi_intent_candidates"]:
        if mi not in valid_intents:
            print(f"  INVALID multi_intent_candidate '{mi}' in {g['candidate_id']}")
            bad_intents += 1
print(f"  Missing fields: {missing}, invalid intent labels: {bad_intents}")

print(f"\n=== H. pair_id uniqueness ===")
pair_ids = [g["pair_id"] for g in golden]
print(f"  Unique: {len(set(pair_ids))} / {len(pair_ids)}")

print(f"\n=== I. Retrieval-corpus leakage guard ===")
Path_out = "data/processed/golden_set_pair_ids_TO_EXCLUDE_FROM_RETRIEVAL.json"
with open(Path_out, "w") as f:
    json.dump(sorted(set(pair_ids)), f, indent=2)
print(f"  Wrote {len(set(pair_ids))} pair_ids to {Path_out}")
print(f"  ACTION REQUIRED when building retrieval corpus: load this file and exclude these pair_ids.")

print(f"\n=== J. Labels changed in this (second) review batch ===")
# recomputed from the correction script's own report: 49 corrected, 4 needs_human_decision, 41 confirmed-no-change
print("  49 corrected, 4 flagged needs_human_decision, 41 confirmed unchanged (see batch2 script output)")

print(f"\n=== K. Total corrections across entire 189-example set ===")
print("  Batch 1 (first 95, now relabeled llm_reviewed_pending_human_signoff): 32 corrected, 63 confirmed")
print("  Batch 2 (remaining 94): 49 corrected, 4 needs_human_decision, 41 confirmed")
print("  TOTAL corrected: 32 + 49 = 81 / 189 (42.9%)")
print("  TOTAL needs_human_decision: 4 / 189 (2.1%)")
print("  TOTAL confirmed unchanged: 63 + 41 = 104 / 189 (55.0%)")

print(f"\n=== L. Examples needing final human decision ===")
for g in golden:
    if g["_review_status"] == "needs_human_decision":
        print(f"  {g['candidate_id']}: {g['customer_text'][:100]}")
        print(f"    -> {g['label_rationale'][:250]}")
