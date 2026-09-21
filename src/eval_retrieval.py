"""
eval_retrieval.py — evaluates the retriever two ways, because exact-answer
relevance can't be measured automatically without inventing labels:

1. AUTOMATIC "same-intent" proxy recall@{1,3,5}, computed over all 188 scored
   golden examples. A retrieved document counts as a "hit" if its
   classifier-predicted intent matches the golden example's TRUE (human-
   reviewed) intent. This is a coarse, explicitly-labeled PROXY for
   relevance -- same topic area, not "this is the right resolution." It's
   cheap to compute at full scale and directional, but it is NOT a
   substitute for real relevance judgment, which is why part 2 exists.

2. MANUAL relevance judgment on a stratified ~30-36 example subset (roughly
   4 per intent). For each, the top-3 retrieved documents (both global and
   intent-aware) are read against the query and judged relevant/not-relevant
   using an explicit rubric:

     RELEVANT = the retrieved historical resolution would give a
     support agent genuinely useful, applicable guidance for the customer's
     actual issue (matching problem type and, where it matters, the
     specific error code / product / defect described) -- not just
     superficial lexical overlap.

   IMPORTANT CAVEAT, consistent with the rest of this project: these
   judgments were made by Claude reading each case against the rubric, not
   by an independent human. This is explicitly flagged as a draft judgment
   pending human spot-check, same caveat that applied to the golden-set
   labels themselves.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from retrieval import Retriever

MENTION_RE = re.compile(r"@\w+")


def load_golden():
    golden = json.loads(open("data/processed/golden_set.json").read())
    return [g for g in golden if not g.get("excluded_from_evaluation", False)]


def automatic_proxy_recall(retriever, golden, mode):
    """Recall@k using same-predicted-intent-as-truth as the relevance proxy."""
    ks = [1, 3, 5]
    hits = {k: 0 for k in ks}
    total = 0
    per_example = []
    for g in golden:
        true_intent = g["intent"]
        results = retriever.retrieve(g["customer_text"], g.get("prior_context", []), k=5, mode=mode)
        retrieved_intents = [r["predicted_intent"] for r in results]
        total += 1
        example_hits = {}
        for k in ks:
            hit = true_intent in retrieved_intents[:k]
            hits[k] += hit
            example_hits[f"hit@{k}"] = hit
        per_example.append({
            "candidate_id": g["candidate_id"],
            "true_intent": true_intent,
            "top5_retrieved_intents": retrieved_intents,
            **example_hits,
        })
    recall = {f"recall@{k}": hits[k] / total for k in ks}
    return recall, per_example


def main():
    print("Loading retriever...")
    retriever = Retriever()
    golden = load_golden()
    print(f"Golden examples scored: {len(golden)}")

    results = {}
    for mode in ["global", "intent_aware"]:
        print(f"\n=== Automatic same-intent proxy recall: {mode} ===")
        recall, per_example = automatic_proxy_recall(retriever, golden, mode)
        for k, v in recall.items():
            print(f"  {k}: {v:.3f}")
        results[mode] = {"proxy_recall": recall, "per_example": per_example}

    Path("data/processed/retrieval_eval_automatic.json").write_text(json.dumps(results, indent=2))
    print("\nSaved data/processed/retrieval_eval_automatic.json")

    # Build the manual-review subset for genuine relevance judgment.
    import random
    rng = random.Random(20260914)
    by_intent = {}
    for g in golden:
        by_intent.setdefault(g["intent"], []).append(g)
    subset = []
    for intent, items in by_intent.items():
        rng.shuffle(items)
        n = min(4, len(items))  # ~4 per intent, ~30-36 total across 9 intents
        subset.extend(items[:n])
    print(f"\nManual review subset size: {len(subset)}")

    manual_dump = []
    for g in subset:
        entry = {
            "candidate_id": g["candidate_id"],
            "true_intent": g["intent"],
            "customer_text": g["customer_text"],
            "prior_context": g.get("prior_context", []),
        }
        for mode in ["global", "intent_aware"]:
            top3 = retriever.retrieve(g["customer_text"], g.get("prior_context", []), k=3, mode=mode)
            entry[f"top3_{mode}"] = [
                {"pair_id": r["pair_id"], "customer_text": r["customer_text"], "brand_text": r["brand_text"],
                 "predicted_intent": r["predicted_intent"], "cosine_sim": r["cosine_sim"]}
                for r in top3
            ]
        manual_dump.append(entry)

    Path("data/processed/retrieval_manual_review_subset.json").write_text(json.dumps(manual_dump, indent=2))
    print("Saved data/processed/retrieval_manual_review_subset.json (for manual relevance judgment)")


if __name__ == "__main__":
    main()
