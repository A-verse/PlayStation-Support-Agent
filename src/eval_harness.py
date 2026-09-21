"""
eval_harness.py — reproducible end-to-end evaluation. Runs the full Agent
(classifier -> retrieval -> reply generation -> safety checks -> escalation)
over the entire golden set and reports every metric requested, with strict
labeling discipline:

  - "proxy" = automatic, coarse, not real relevance/correctness
  - "LLM-assisted" / "Claude-judged" = produced by Claude reading examples
    against a rubric, NOT an independent human -- this describes BOTH the
    golden-set intent/escalation labels (status: llm_reviewed_pending_human_
    signoff, see golden_set.json's _review_status field) and the retrieval
    relevance judgments from the retrieval stage.
  - "human-verified" is used ONLY if independent human annotations exist.
    As of this run, they do not -- the golden set is explicitly pending the
    user's sign-off. Every place this harness reports escalation P/R/F1
    "against golden labels", the golden labels themselves carry that same
    caveat, and this is restated in the output, not just here.

This script does not re-run classifier training or retrieval-corpus
construction (those are separate, already-reproducible stages) -- it
reuses their saved artifacts and adds full end-to-end Agent execution and
escalation-specific metrics, which didn't exist yet.
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from agent import Agent


def load_golden():
    golden = json.loads(open("data/processed/golden_set.json").read())
    scored = [g for g in golden if not g.get("excluded_from_evaluation", False)]
    return scored, len(golden) - len(scored)


def escalation_prf(predictions, ground_truth):
    """predictions/ground_truth: lists of bool (True=escalate)."""
    tp = sum(1 for p, g in zip(predictions, ground_truth) if p and g)
    fp = sum(1 for p, g in zip(predictions, ground_truth) if p and not g)
    fn = sum(1 for p, g in zip(predictions, ground_truth) if not p and g)
    tn = sum(1 for p, g in zip(predictions, ground_truth) if not p and not g)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(predictions) if predictions else 0.0
    return {
        "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "false_escalation_rate": fp / (tp + fp) if (tp + fp) else 0.0,  # of predicted escalations, how many were unnecessary
        "false_auto_handle_rate": fn / (fn + tn) if (fn + tn) else 0.0,  # of predicted auto-handles, how many should have escalated
    }


def main():
    golden, n_excluded = load_golden()
    print(f"Golden examples scored: {len(golden)} (excluded_from_evaluation: {n_excluded})")
    print("\n" + "=" * 78)
    print("SECTION 1: INTENT CLASSIFICATION (reused from classifier stage, unchanged)")
    print("=" * 78)
    clf_results = json.loads(open("data/processed/classifier_eval_results.json").read())
    tfidf = clf_results["results"]["tfidf_logreg"]
    maj = clf_results["results"]["majority_baseline"]
    print(f"NOTE: trained on {clf_results['training_size']:,} WEAK/SILVER labels (regex heuristic), "
          f"NOT hand-labeled ground truth. Evaluated against the {clf_results['golden_eval_size']}-example "
          f"golden set (LLM-reviewed, pending human sign-off -- see caveat below).")
    print(f"Majority baseline:   accuracy={maj['accuracy']:.3f}  macro_f1={maj['macro_f1']:.3f}")
    print(f"TF-IDF + LogReg:     accuracy={tfidf['accuracy']:.3f}  macro_f1={tfidf['macro_f1']:.3f}  weighted_f1={tfidf['weighted_f1']:.3f}")

    print("\n" + "=" * 78)
    print("SECTION 2: RETRIEVAL (reused from retrieval stage, unchanged)")
    print("=" * 78)
    retr = json.loads(open("data/processed/retrieval_eval_automatic.json").read())
    manual = json.loads(open("data/processed/retrieval_manual_relevance_results.json").read())
    print("PROXY recall (same-intent-label match, n=188, automatic, NOT true relevance):")
    for mode in ["global", "intent_aware"]:
        r = retr[mode]["proxy_recall"]
        print(f"  {mode:<14} recall@1={r['recall@1']:.3f} recall@3={r['recall@3']:.3f} recall@5={r['recall@5']:.3f}")
    print(f"\nCLAUDE-JUDGED relevance (n=36 subset, NOT independent human validation -- {manual['methodology'][:80]}...):")
    for mode in ["global", "intent_aware"]:
        r1 = manual["results"]["manual_recall_at_1"][mode]
        r3 = manual["results"]["manual_recall_at_3_binary_at_least_one_relevant"][mode]
        print(f"  {mode:<14} recall@1={r1:.3f} recall@3(>=1 relevant)={r3:.3f}")

    print("\n" + "=" * 78)
    print("SECTION 3: FULL PIPELINE RUN (new -- Agent over all golden examples)")
    print("=" * 78)
    agent = Agent()
    print(f"Mock mode: {agent.reply_generator.mock_mode} (no ANTHROPIC_API_KEY in this environment)")

    full_results = []
    for g in golden:
        out = agent.handle(g["customer_text"], g.get("prior_context", []))
        out["candidate_id"] = g["candidate_id"]
        out["golden_true_intent"] = g["intent"]
        out["golden_should_escalate"] = g["should_escalate"]  # LLM-reviewed, pending human sign-off -- see caveat
        out["golden_out_of_scope"] = g.get("out_of_scope", False)
        full_results.append(out)

    Path("data/processed/eval_full_pipeline_results.json").write_text(json.dumps(full_results, indent=2, default=str))
    print(f"Wrote full per-example pipeline results to data/processed/eval_full_pipeline_results.json")

    n_evidence_failure = sum(1 for r in full_results if r["evidence_failure"])
    n_safety_failure = sum(1 for r in full_results if r["response_safety_failure"])
    action_counts = Counter(r["escalation_decision"]["action"] for r in full_results)
    reason_counts = Counter()
    for r in full_results:
        for code in r["escalation_decision"]["reason_codes"]:
            reason_counts[code] += 1

    print(f"\nevidence_failure rate: {n_evidence_failure}/{len(full_results)} ({n_evidence_failure/len(full_results):.1%})")
    print(f"response_safety_failure rate: {n_safety_failure}/{len(full_results)} ({n_safety_failure/len(full_results):.1%})")
    print(f"\nAuto-handle vs escalate:")
    for action, n in action_counts.items():
        print(f"  {action:<12} {n:>4}  ({n/len(full_results):.1%})")
    print(f"\nReason-code frequency (a decision can have multiple):")
    for code, n in reason_counts.most_common():
        print(f"  {code:<40} {n:>4}")

    print("\n" + "=" * 78)
    print("SECTION 4: ESCALATION METRICS vs GOLDEN should_escalate LABELS")
    print("=" * 78)
    print("*** CAVEAT: golden_set.json's should_escalate field is itself LLM-reviewed,")
    print("*** pending the user's human sign-off (see _review_status per example).")
    print("*** These are NOT precision/recall against human-verified ground truth.")

    predictions = [r["escalation_decision"]["action"] == "escalate" for r in full_results]
    ground_truth = [r["golden_should_escalate"] for r in full_results]

    engine_metrics = escalation_prf(predictions, ground_truth)
    always_auto = escalation_prf([False] * len(full_results), ground_truth)
    always_escalate = escalation_prf([True] * len(full_results), ground_truth)

    print(f"\n{'Approach':<24}{'Acc':>7}{'Prec':>7}{'Rec':>7}{'F1':>7}{'FalseEsc%':>11}{'FalseAuto%':>11}")
    for name, m in [("Trivial: always auto", always_auto), ("Trivial: always escalate", always_escalate), ("Escalation engine", engine_metrics)]:
        print(f"{name:<24}{m['accuracy']:>7.2f}{m['precision']:>7.2f}{m['recall']:>7.2f}{m['f1']:>7.2f}{m['false_escalation_rate']:>11.2%}{m['false_auto_handle_rate']:>11.2%}")

    print("\n" + "=" * 78)
    print("SECTION 5: LLM-AS-JUDGE")
    print("=" * 78)
    print("NOT RUN. No ANTHROPIC_API_KEY available in this environment. Reply generation")
    print("ran in mock/template mode (see reply_generator.py), so an LLM-judge pass over")
    print("mock-mode replies would mostly be judging template selection, not language")
    print("quality. Recommend running this once the real LLM path is enabled with a key.")

    print("\n" + "=" * 78)
    print("SECTION 6: HUMAN-REVIEW AGREEMENT")
    print("=" * 78)
    print("NOT AVAILABLE. No independent human annotations exist yet for this project --")
    print("the golden set's labels were produced by Claude (LLM-assisted), explicitly")
    print("flagged _review_status='llm_reviewed_pending_human_signoff' throughout")
    print("golden_set.json, pending the user's sign-off. See REVIEW_GUIDE.md for the")
    print("practical workflow to review/correct labels.")

    summary = {
        "golden_eval_size": len(golden),
        "golden_excluded_from_evaluation": n_excluded,
        "intent_classification": {"majority_baseline": maj, "tfidf_logreg": tfidf,
                                    "training_label_caveat": "weak/silver labels, not hand-labeled"},
        "retrieval": {"automatic_proxy": {m: retr[m]["proxy_recall"] for m in ["global", "intent_aware"]},
                      "claude_judged_subset_n36": manual["results"],
                      "caveat": "proxy = same-intent-label match, not true relevance. Claude-judged subset is NOT independent human validation."},
        "pipeline_run": {
            "evidence_failure_rate": n_evidence_failure / len(full_results),
            "response_safety_failure_rate": n_safety_failure / len(full_results),
            "action_counts": dict(action_counts),
            "reason_code_counts": dict(reason_counts),
        },
        "escalation_vs_baselines": {
            "always_auto_handle": always_auto,
            "always_escalate": always_escalate,
            "escalation_engine": engine_metrics,
            "caveat": "ground truth (golden should_escalate) is LLM-reviewed, pending human sign-off -- not human-verified ground truth",
        },
        "llm_as_judge": "NOT RUN -- no ANTHROPIC_API_KEY available",
        "human_review_agreement": "NOT AVAILABLE -- no independent human annotations exist yet",
    }
    Path("data/processed/eval_harness_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote summary to data/processed/eval_harness_summary.json")


if __name__ == "__main__":
    main()
