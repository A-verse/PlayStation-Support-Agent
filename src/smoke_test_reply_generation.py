"""
smoke_test_reply_generation.py — runs the reply generator over a diverse
sample of golden-set examples (mock mode, since no ANTHROPIC_API_KEY is set
in this environment) and saves full per-example results for review.
"""

import json
import random
from pathlib import Path

from reply_generator import ReplyGenerator


def main():
    golden = json.loads(open("data/processed/golden_set.json").read())
    golden = [g for g in golden if not g.get("excluded_from_evaluation", False)]

    # Diverse sample: some followups, some not, spread across intents, plus
    # anything flagged out_of_scope, plus a few explicit "already tried" cases.
    rng = random.Random(20260914)
    by_intent = {}
    for g in golden:
        by_intent.setdefault(g["intent"], []).append(g)
    sample = []
    for intent, items in by_intent.items():
        rng.shuffle(items)
        sample.extend(items[:5])

    print(f"Running reply generation over {len(sample)} examples...")
    gen = ReplyGenerator()
    print(f"Mock mode: {gen.mock_mode}\n")

    results = []
    for g in sample:
        out = gen.generate(g["customer_text"], g.get("prior_context", []))
        out["candidate_id"] = g["candidate_id"]
        out["true_intent"] = g["intent"]
        out["is_followup"] = g.get("is_followup", False)
        out["out_of_scope"] = g.get("out_of_scope", False)
        results.append(out)

    Path("data/processed/reply_generation_smoke_test.json").write_text(json.dumps(results, indent=2))
    print(f"Wrote {len(results)} results to data/processed/reply_generation_smoke_test.json")

    n_evidence_failure = sum(1 for r in results if r["evidence_failure"])
    n_already_tried = sum(1 for r in results if r["already_tried_detected"])
    print(f"evidence_failure: {n_evidence_failure}/{len(results)}")
    print(f"already_tried_detected: {n_already_tried}/{len(results)}")


if __name__ == "__main__":
    main()
