"""
train_and_eval_classifier.py — trains the majority-class trivial baseline and
the TF-IDF + Logistic Regression simple baseline, evaluates both against the
189-example hand-reviewed golden set (188 scored; 1 excluded_from_evaluation
per approved policy), and saves all artifacts.

Reproducibility: fixed random_state throughout; run this script top-to-bottom
on the same pairs.jsonl / golden_set.json and you get identical numbers.

Leakage: training_set.jsonl was built by build_training_set.py with an
explicit assertion that none of the 189 golden pair_ids appear in it. This
script re-asserts that at load time as a second safety check.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score)
from sklearn.pipeline import Pipeline

MENTION_RE = re.compile(r"@\w+")
RANDOM_STATE = 42

INTENT_LABELS = [
    "Account Access & Recovery",
    "Purchases, Billing & Refunds",
    "Error Codes & Technical Faults",
    "Network & Service Connectivity",
    "Console Troubleshooting (Safe Mode / General Fix)",
    "Downloads & Digital Content",
    "PlayStation Vue / Streaming Service",
    "Hardware / Device Malfunction",
    "Unclear / Insufficient Information",
]


def enrich(g):
    ctx_text = " ".join(c["text"] for c in g.get("prior_context", []))
    text = g.get("text") if "text" in g else g.get("customer_text")
    return MENTION_RE.sub(" ", f"{ctx_text} {text}").strip()


def load_training():
    rows = [json.loads(l) for l in open("data/processed/training_set.jsonl")]
    exclude_ids = set(json.loads(open("data/processed/golden_set_pair_ids_TO_EXCLUDE_FROM_RETRIEVAL.json").read()))
    assert exclude_ids.isdisjoint(r["pair_id"] for r in rows), "LEAKAGE: a golden pair_id is in training_set.jsonl!"
    X = [r["text"] for r in rows]
    y = [r["label"] for r in rows]
    return X, y


def load_golden():
    golden = json.loads(open("data/processed/golden_set.json").read())
    scored = [g for g in golden if not g.get("excluded_from_evaluation", False)]
    excluded = len(golden) - len(scored)
    X = [enrich(g) for g in scored]
    y = [g["intent"] for g in scored]
    meta = scored
    return X, y, meta, excluded


def evaluate(y_true, y_pred, name):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=INTENT_LABELS, zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", labels=INTENT_LABELS, zero_division=0)
    report = classification_report(
        y_true, y_pred, labels=INTENT_LABELS, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=INTENT_LABELS)

    print(f"\n=== {name} ===")
    print(f"Accuracy:    {acc:.3f}")
    print(f"Macro F1:    {macro_f1:.3f}")
    print(f"Weighted F1: {weighted_f1:.3f}")
    print(f"\n{'Intent':<55} {'Prec':>6} {'Rec':>6} {'F1':>6} {'N':>5}")
    for intent in INTENT_LABELS:
        r = report[intent]
        print(f"{intent:<55} {r['precision']:>6.2f} {r['recall']:>6.2f} {r['f1-score']:>6.2f} {int(r['support']):>5}")

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_intent": {intent: report[intent] for intent in INTENT_LABELS},
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": INTENT_LABELS,
    }


def main():
    print("Loading training data (weak/silver labels, golden-excluded)...")
    X_train, y_train = load_training()
    print(f"Training examples: {len(X_train):,}")
    print(f"Training label distribution: {Counter(y_train).most_common()}")

    print("\nLoading golden evaluation set...")
    X_gold, y_gold, gold_meta, n_excluded = load_golden()
    print(f"Golden examples scored: {len(X_gold)} (excluded_from_evaluation: {n_excluded})")

    results = {}

    # --- 1. Trivial baseline: majority class ---
    majority_clf = DummyClassifier(strategy="most_frequent")
    majority_clf.fit(X_train, y_train)
    y_pred_majority = majority_clf.predict(X_gold)
    results["majority_baseline"] = evaluate(y_gold, y_pred_majority, "TRIVIAL BASELINE: Majority Class")
    results["majority_baseline"]["predicted_class"] = Counter(y_train).most_common(1)[0][0]

    # --- 2. Simple baseline: TF-IDF + Logistic Regression ---
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=2, max_df=0.5, stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE, C=1.0)),
    ])
    pipeline.fit(X_train, y_train)
    y_pred_tfidf = pipeline.predict(X_gold)
    results["tfidf_logreg"] = evaluate(y_gold, y_pred_tfidf, "SIMPLE BASELINE: TF-IDF + Logistic Regression")

    # --- Save artifacts ---
    Path("models").mkdir(exist_ok=True)
    joblib.dump(pipeline, "models/tfidf_logreg_intent_classifier.joblib")
    joblib.dump(majority_clf, "models/majority_baseline.joblib")
    print("\nSaved models/tfidf_logreg_intent_classifier.joblib")
    print("Saved models/majority_baseline.joblib")

    # per-example predictions for error analysis later
    predictions_dump = []
    for i, g in enumerate(gold_meta):
        predictions_dump.append({
            "candidate_id": g["candidate_id"],
            "customer_text": g["customer_text"],
            "true_intent": y_gold[i],
            "majority_pred": y_pred_majority[i],
            "tfidf_pred": y_pred_tfidf[i],
            "correct_tfidf": y_pred_tfidf[i] == y_gold[i],
            "is_followup": g.get("is_followup", False),
            "out_of_scope": g.get("out_of_scope", False),
        })
    Path("data/processed/classifier_predictions.json").write_text(json.dumps(predictions_dump, indent=2))

    Path("data/processed/classifier_eval_results.json").write_text(json.dumps({
        "training_size": len(X_train),
        "golden_eval_size": len(X_gold),
        "golden_excluded_from_evaluation": n_excluded,
        "results": results,
    }, indent=2, default=str))

    print("\nSaved data/processed/classifier_predictions.json")
    print("Saved data/processed/classifier_eval_results.json")

    # Highlight the 4 specifically requested intents
    print("\n=== Requested spotlight: Hardware, Vue, Downloads, Unclear ===")
    for intent in ["Hardware / Device Malfunction", "PlayStation Vue / Streaming Service",
                   "Downloads & Digital Content", "Unclear / Insufficient Information"]:
        maj = results["majority_baseline"]["per_intent"][intent]
        tf = results["tfidf_logreg"]["per_intent"][intent]
        print(f"\n{intent} (n={int(tf['support'])}):")
        print(f"  Majority baseline: P={maj['precision']:.2f} R={maj['recall']:.2f} F1={maj['f1-score']:.2f}")
        print(f"  TF-IDF+LogReg:     P={tf['precision']:.2f} R={tf['recall']:.2f} F1={tf['f1-score']:.2f}")


if __name__ == "__main__":
    main()
