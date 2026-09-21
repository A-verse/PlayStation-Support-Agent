import json
from collections import Counter

golden = json.loads(open("data/processed/golden_set_v1.json").read())

print(f"Total examples: {len(golden)}\n")

print("=== Final intent distribution (post manual correction) ===")
intent_counts = Counter(g["intent"] for g in golden)
for intent, n in sorted(intent_counts.items(), key=lambda x: -x[1]):
    print(f"  {intent:<55} {n:>4}  ({n/len(golden):.1%})")

print(f"\nmulti_intent_candidates non-empty: {sum(1 for g in golden if g['multi_intent_candidates'])}")
print(f"is_followup: {sum(1 for g in golden if g['is_followup'])}")
print(f"should_escalate=True: {sum(1 for g in golden if g['should_escalate'])}")
print(f"review_status manually_verified: {sum(1 for g in golden if g['_review_status']=='manually_verified')}")
print(f"review_status rule_based_draft (pending real human review): {sum(1 for g in golden if g['_review_status']=='rule_based_draft')}")

print("\n=== Duplicate / near-duplicate check ===")
seen = {}
dupes = 0
for g in golden:
    key = g["customer_text"].lower().strip()
    if key in seen:
        dupes += 1
        print(f"  DUPLICATE: {g['candidate_id']} == {seen[key]}: {g['customer_text'][:80]}")
    seen[key] = g["candidate_id"]
print(f"Exact-duplicate customer_text pairs found: {dupes}")

pair_ids = [g["pair_id"] for g in golden]
print(f"Unique pair_ids: {len(set(pair_ids))} / {len(pair_ids)} (should be equal)")

print("\n=== Length / difficulty coverage ===")
lens = [g["word_count"] for g in golden]
print(f"Word count: min={min(lens)} median={sorted(lens)[len(lens)//2]} max={max(lens)}")
print(f"Short (<=4 words): {sum(1 for l in lens if l<=4)}")
print(f"Long (>=30 words): {sum(1 for l in lens if l>=30)}")

print("\n=== Structural completeness ===")
required_fields = ["customer_text","prior_context","intent","multi_intent_candidates",
                    "should_escalate","expected_action","label_rationale"]
missing = 0
for g in golden:
    for f in required_fields:
        if f not in g:
            print(f"  MISSING FIELD {f} in {g['candidate_id']}")
            missing += 1
print(f"Missing required fields: {missing}")
