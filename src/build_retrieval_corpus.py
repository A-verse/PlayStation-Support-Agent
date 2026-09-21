"""
build_retrieval_corpus.py — builds the retrieval corpus of historical
(customer_message [+ context], brand_reply) pairs, explicitly excluding all
189 golden-set pair_ids (verified programmatically, not just assumed).

Each corpus document also gets a `predicted_intent` tag from the trained
TF-IDF+LogReg classifier (not the raw regex proxy) -- the classifier
generalizes a bit better than the regex it was trained on, and having intent
tags on the retrieval corpus is what makes intent-aware retrieval possible in
the next stage. This is documented as classifier-predicted, not ground
truth -- the same weak-label caveat that applies to the classifier itself
carries through to these tags.
"""

import json
import re
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).parent))

MENTION_RE = re.compile(r"@\w+")


def enrich(p):
    ctx_text = " ".join(c["text"] for c in p.get("prior_context", []))
    return MENTION_RE.sub(" ", f"{ctx_text} {p['customer_text']}").strip()


def main():
    pairs = [json.loads(l) for l in open("data/processed/pairs.jsonl")]
    exclude_ids = set(json.loads(open("data/processed/golden_set_pair_ids_TO_EXCLUDE_FROM_RETRIEVAL.json").read()))

    corpus = [p for p in pairs if p["pair_id"] not in exclude_ids]
    print(f"Total pairs: {len(pairs):,}")
    print(f"Excluded (golden leakage guard): {len(pairs) - len(corpus):,} (expected 189)")
    print(f"Retrieval corpus size: {len(corpus):,}")

    assert len(pairs) - len(corpus) == len(exclude_ids), "Leakage guard count mismatch!"
    corpus_ids = set(p["pair_id"] for p in corpus)
    assert corpus_ids.isdisjoint(exclude_ids), "LEAKAGE DETECTED: a golden pair_id is in the retrieval corpus!"
    print("Verified: zero golden pair_ids present in retrieval corpus.")

    clf = joblib.load("models/tfidf_logreg_intent_classifier.joblib")
    texts = [enrich(p) for p in corpus]
    predicted_intents = clf.predict(texts)

    docs = []
    for p, text, intent in zip(corpus, texts, predicted_intents):
        docs.append({
            "pair_id": p["pair_id"],
            "query_text": text,  # enriched (context + customer_text), used for indexing
            "customer_text": p["customer_text"],
            "prior_context": p["prior_context"],
            "brand_text": p["brand_text"],  # the historical resolution
            "predicted_intent": intent,  # from the trained classifier, NOT ground truth
            "created_at": p["created_at"],
        })

    Path("data/processed/retrieval_corpus.jsonl").write_text(
        "\n".join(json.dumps(d) for d in docs) + "\n"
    )
    print(f"\nWrote {len(docs):,} documents to data/processed/retrieval_corpus.jsonl")

    from collections import Counter
    dist = Counter(d["predicted_intent"] for d in docs)
    print("\nPredicted-intent distribution in retrieval corpus:")
    for intent, n in dist.most_common():
        print(f"  {intent:<55} {n:>6,}  ({n/len(docs):.1%})")


if __name__ == "__main__":
    main()
