"""
build_training_set.py — builds the training set for the TF-IDF baseline
classifier.

IMPORTANT, stated plainly: these training labels are WEAK/SILVER labels from
the same regex heuristic used for golden-set sampling stratification
(intent_signals.proxy_intent), NOT hand-labeled ground truth. The 189
hand-reviewed golden examples are reserved entirely for evaluation and must
never appear in training. This is a deliberate, documented simplification
(see decision_log.md) -- hand-labeling 15,806 examples was never in scope,
and training on regex-heuristic labels is a legitimate, transparent baseline
approach as long as it's not mistaken for genuine supervision. Expect the
classifier to inherit the regex's known systematic errors (e.g. the false
positives found during golden-set review), which is itself a useful,
honest thing to show in the evaluation report.

Context representation: training text = prior_context (all turns, either
role) + customer_text, same enrichment used in explore_intents.py, so the
~30% of examples that are follow-ups get real topical signal instead of
training on a context-free fragment.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from intent_signals import proxy_intent

MENTION_RE = re.compile(r"@\w+")


def enrich(p):
    ctx_text = " ".join(c["text"] for c in p.get("prior_context", []))
    return MENTION_RE.sub(" ", f"{ctx_text} {p['customer_text']}").strip()


def main():
    pairs = [json.loads(l) for l in open("data/processed/pairs.jsonl")]
    exclude_ids = set(json.loads(open("data/processed/golden_set_pair_ids_TO_EXCLUDE_FROM_RETRIEVAL.json").read()))

    print(f"Total pairs: {len(pairs):,}")
    print(f"Golden pair_ids to exclude: {len(exclude_ids)}")

    training = []
    excluded_count = 0
    for p in pairs:
        if p["pair_id"] in exclude_ids:
            excluded_count += 1
            continue
        text = enrich(p)
        label = proxy_intent(text)
        training.append({"pair_id": p["pair_id"], "text": text, "label": label})

    print(f"Excluded (golden leakage guard): {excluded_count} (expected 189)")
    print(f"Training examples: {len(training):,}")

    assert excluded_count == len(exclude_ids), "Leakage guard mismatch -- not all golden pair_ids were found in the corpus!"
    assert exclude_ids.isdisjoint(p["pair_id"] for p in training), "LEAKAGE DETECTED: a golden pair_id ended up in training data!"

    from collections import Counter
    label_counts = Counter(t["label"] for t in training)
    print("\nWeak-label distribution (training):")
    for label, n in label_counts.most_common():
        print(f"  {label:<55} {n:>6,}  ({n/len(training):.1%})")

    Path("data/processed/training_set.jsonl").write_text(
        "\n".join(json.dumps(t) for t in training) + "\n"
    )
    print(f"\nWrote {len(training):,} training examples to data/processed/training_set.jsonl")


if __name__ == "__main__":
    main()
