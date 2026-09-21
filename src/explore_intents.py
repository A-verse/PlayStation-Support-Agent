"""
explore_intents.py — EXPLORATORY ONLY. Not the final classifier.

Purpose: generate evidence for a human (me, in this case) to derive an intent
taxonomy bottom-up from the actual data, rather than assuming generic
"PlayStation support categories."

Two things happen here:
1. A large random sample of raw customer messages is dumped for direct manual
   reading (this is the ground truth -- clustering below is just a navigation
   aid over 15,995 messages, not a substitute for reading real examples).
2. TF-IDF + KMeans clustering over ALL 15,995 English pairs, with top terms
   and nearest-to-centroid examples per cluster, to see the data's natural
   groupings and get defensible frequency estimates once clusters are
   manually merged into named intents.

Deliberately simple: TF-IDF + KMeans, no embeddings API, no external model
download (sandbox network doesn't reach model hubs anyway). Fully
deterministic given a fixed random_state.
"""

import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

MENTION_RE = re.compile(r"@\w+")


def load_pairs(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def strip_mentions(text):
    return MENTION_RE.sub(" ", text).strip()


def main():
    pairs = load_pairs("data/processed/pairs.jsonl")
    print(f"Loaded {len(pairs):,} English pairs\n")

    texts_raw = [p["customer_text"] for p in pairs]
    # Enriched text: fold in prior context (both roles) so follow-up turns like
    # "still doesn't work" get real topical signal instead of clustering as noise.
    def enrich(p):
        ctx_text = " ".join(c["text"] for c in p.get("prior_context", []))
        return f"{ctx_text} {p['customer_text']}".strip()

    texts_enriched = [enrich(p) for p in pairs]
    texts_clean = [strip_mentions(t) for t in texts_enriched]

    # ---- 1. Large random sample for manual reading ----
    random.seed(11)
    sample_idx = random.sample(range(len(pairs)), 300)
    with open("data/processed/manual_reading_sample.txt", "w") as f:
        for i in sample_idx:
            f.write(f"{i}\t{texts_raw[i]}\n")
    print("Wrote 300-message random sample to data/processed/manual_reading_sample.txt")

    # ---- 2. TF-IDF + KMeans over the full set ----
    vectorizer = TfidfVectorizer(
        max_features=4000,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=5,
        max_df=0.4,
    )
    X = vectorizer.fit_transform(texts_clean)
    terms = np.array(vectorizer.get_feature_names_out())

    k = 22
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    print(f"\nClustered {len(pairs):,} messages into k={k} clusters (TF-IDF + KMeans)\n")
    print("=" * 90)

    cluster_report = []
    for c in range(k):
        idxs = np.where(labels == c)[0]
        size = len(idxs)
        if size == 0:
            continue
        centroid = km.cluster_centers_[c]
        top_terms = terms[np.argsort(centroid)[::-1][:12]]

        # nearest-to-centroid examples (more representative than random within cluster)
        cluster_X = X[idxs].toarray()
        dists = np.linalg.norm(cluster_X - centroid, axis=1)
        nearest = idxs[np.argsort(dists)[:6]]

        examples = [texts_raw[i] for i in nearest]
        print(f"CLUSTER {c}  (n={size}, {size/len(pairs):.1%})")
        print(f"  top terms: {', '.join(top_terms)}")
        for ex in examples:
            print(f"    - {ex[:110]}")
        print()

        cluster_report.append(
            {"cluster": c, "size": size, "pct": size / len(pairs), "top_terms": top_terms.tolist(), "examples": examples}
        )

    Path("data/processed/intent_clusters.json").write_text(json.dumps(cluster_report, indent=2))
    print("Wrote full cluster report to data/processed/intent_clusters.json")


if __name__ == "__main__":
    main()
