"""
retrieval.py — simple TF-IDF cosine-similarity retriever over the historical
(customer_message [+ context], brand_reply) corpus.

Two modes:
  - global: plain cosine similarity ranking, no intent signal.
  - intent_aware: cosine similarity + a bonus for corpus documents whose
    classifier-predicted intent matches the QUERY's predicted intent. This is
    a soft rerank, not a hard filter -- a hard filter would zero out recall
    whenever the classifier's intent prediction is wrong (24% of the time per
    the classifier eval), which seemed like an obviously bad trade before
    even measuring it. Reranking is more robust: intent-matching documents
    get a score boost, but a highly-similar wrong-intent document can still
    win if the text match is strong enough.

Kept deliberately simple: TF-IDF + cosine similarity, in-memory, no vector
database, no ANN index. At ~15.8k documents this runs in milliseconds and a
vector DB would add machinery with no measurable benefit at this scale.
"""

import json
import re
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MENTION_RE = re.compile(r"@\w+")
INTENT_MATCH_BONUS = 0.15  # additive bonus on cosine similarity (0-1 scale) for matching predicted intent


def enrich(customer_text, prior_context):
    ctx_text = " ".join(c["text"] for c in prior_context)
    return MENTION_RE.sub(" ", f"{ctx_text} {customer_text}").strip()


class Retriever:
    def __init__(self, corpus_path="data/processed/retrieval_corpus.jsonl"):
        self.docs = [json.loads(l) for l in open(corpus_path)]
        self.vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=2, max_df=0.5, stop_words="english")
        self.doc_matrix = self.vectorizer.fit_transform([d["query_text"] for d in self.docs])
        self.intent_clf = joblib.load("models/tfidf_logreg_intent_classifier.joblib")

    def predict_query_intent(self, query_text):
        return self.intent_clf.predict([query_text])[0]

    def retrieve(self, customer_text, prior_context=None, k=5, mode="global", query_intent=None):
        """mode: 'global' or 'intent_aware'. If mode='intent_aware' and
        query_intent is None, the query's intent is predicted on the fly
        with the same classifier used to tag the corpus."""
        prior_context = prior_context or []
        query_text = enrich(customer_text, prior_context)
        query_vec = self.vectorizer.transform([query_text])
        sims = cosine_similarity(query_vec, self.doc_matrix)[0]

        if mode == "intent_aware":
            if query_intent is None:
                query_intent = self.predict_query_intent(query_text)
            bonus = np.array([INTENT_MATCH_BONUS if d["predicted_intent"] == query_intent else 0.0 for d in self.docs])
            scores = sims + bonus
        elif mode == "global":
            scores = sims
        else:
            raise ValueError(f"unknown mode {mode}")

        top_idx = np.argsort(scores)[::-1][:k]
        results = []
        for rank, idx in enumerate(top_idx):
            d = self.docs[idx]
            results.append({
                "rank": rank + 1,
                "cosine_sim": float(sims[idx]),
                "score": float(scores[idx]),
                "pair_id": d["pair_id"],
                "customer_text": d["customer_text"],
                "prior_context": d["prior_context"],
                "brand_text": d["brand_text"],
                "predicted_intent": d["predicted_intent"],
            })
        return results

    def save(self, path="models/retriever_vectorizer.joblib"):
        joblib.dump({"vectorizer": self.vectorizer, "doc_matrix": self.doc_matrix}, path)


if __name__ == "__main__":
    r = Retriever()
    r.save()
    print(f"Retriever built over {len(r.docs):,} documents.")
    print("Saved vectorizer+doc_matrix to models/retriever_vectorizer.joblib")

    # smoke test
    demo = r.retrieve("my ps4 controller won't charge and turns off randomly", k=3, mode="global")
    print("\nDemo query: \"my ps4 controller won't charge and turns off randomly\"")
    for res in demo:
        print(f"  [{res['rank']}] sim={res['cosine_sim']:.3f} intent={res['predicted_intent']}")
        print(f"      Q: {res['customer_text'][:90]}")
        print(f"      A: {res['brand_text'][:90]}")
