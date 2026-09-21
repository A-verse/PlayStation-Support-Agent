# Report — AskPlayStation Support Agent

## 1. Problem framing
Build an AI support agent for one brand from the Kaggle Twitter customer-support
dataset that classifies intent, drafts a grounded reply from historical resolutions,
and decides auto-handle vs. escalate with a stated reason — evaluated honestly, not
just built. AskPlayStation was chosen for having the lowest "please DM us" deflection
rate among viable high-volume candidates (25.3%), meaning its historical replies
disproportionately contain actual troubleshooting content rather than deflections.

## 2. What "good" means for this brand
Not "high accuracy" in isolation. Specifically:
- A classifier whose **per-intent** performance is visible, since 9 unevenly-sized
  intents make aggregate accuracy misleading on its own.
- Retrieval that surfaces genuinely applicable historical fixes, not just lexically
  similar text — validated with a real (if small) manual relevance check, not just an
  automatic proxy.
- A reply generator that **never invents policy** and fails safe (asks a clarifying
  question) rather than confidently answering with weak evidence.
- An escalation policy where a **false auto-handle** (confidently wrong) is treated as
  worse than a **false escalation** (unnecessarily routed to a human) — but where
  customer frustration alone never triggers escalation on its own.

## 3. What we deliberately chose NOT to build
No vector database (TF-IDF cosine over ~15.8k docs runs in milliseconds), no
multi-agent orchestration, no fine-tuning, no LLM reranker (added only where a simple
heuristic demonstrably failed, and even then reused the retriever's existing TF-IDF
IDF weights rather than a new model), no production infra (queues, microservices), no
processing of the full 3M-row dataset. Reply generation itself defaults to a
deterministic template mode specifically so the whole project runs without an API key.

## 4. Results against two baselines

**Intent classification:**
| | Accuracy | Macro F1 |
|---|---|---|
| Trivial (majority class) | 6.4% | 0.013 |
| Simple (TF-IDF + Logistic Regression) | 76.1% | 0.760 |

**Escalation decision** (ground truth: golden set's `should_escalate`, itself
LLM-reviewed pending human sign-off — see caveat in Section 6):
| | Accuracy | F1 | False-escalate % | False-auto-handle % |
|---|---|---|---|---|
| Trivial (always auto-handle) | 0.63 | 0.00 | 0.0% | 36.7% |
| Trivial (always escalate) | 0.37 | 0.54 | 63.3% | 0.0% |
| Escalation engine | 0.70 | 0.55 | 38.2% | 26.3% |

## 5. Top 5 failure modes (real, from actual experiments)

**1. Rare intents are lexically scattered, not clustered.** Hardware complaints
(1.9% natural frequency) use wildly different phrasings ("blue light of death",
"disc won't eject", "controller won't charge") that don't form one coherent lexical
cluster. *Example:* the classifier gets 0.94 precision but only 0.62 recall on
Hardware — confident when right, but misses many real cases. *Hypothesis:* TF-IDF
has no way to generalize across semantically-related-but-lexically-different
phrasings without embeddings. *Improvement:* targeted data augmentation for rare
intents, or a small embedding-based fallback specifically for low-confidence cases.

**2. "Unclear" is precision-poor for the classifier.** 0.42 precision, 0.83 recall —
the model over-predicts Unclear, catching real cases but also wrongly dumping some
Account/Network messages into it. *Example:* short account-related messages
sometimes get misclassified as Unclear. *Hypothesis:* Unclear's definition (message
quality, not topic) is fundamentally a different kind of signal than the other 8
intents, and a single classifier trained the same way for both is a mismatch.
*Improvement:* a two-stage classifier — first "is this message classifiable at all,"
then topic classification only if yes.

**3. Retrieval finds boilerplate dead ends.** Some error codes' entire historical
reply population is "please check your DMs," with zero visible troubleshooting
content. *Example:* `cand_0153` ("bricked my system") — all top-3 retrieved replies
are content-free DM deflections. *Hypothesis:* this specific error is inherently
account-specific, so the brand always privately DMs; no public reply ever had
content to learn from. *Improvement:* explicitly flag error codes/intents with a high
DM-deflection rate in their historical population and route them straight to
escalation rather than attempting grounded generation.

**4. Context-dependent follow-ups need context, and context can also mislead.**
Fixed a real bug where prior conversation context wasn't threading through at all
(decision log #5), then found the fix could over-include boilerplate context and
mask real mismatches (decision log #16, the `cand_0035` regression during safety-
check iteration). *Improvement:* the IDF-weighted overlap check partially addresses
this; a proper fix would need better context-salience weighting, not just IDF.

**5. Simple heuristic safety checks trade false positives for false negatives.**
Every fix to the response-safety checks (curly-apostrophe normalization, context
folding, stemming, IDF-weighting) fixed specific known failures but introduced new
ones elsewhere (decision log #15–16). *Example:* fixing `cand_0035` came at the cost
of new false positives on `cand_0052`/`cand_0077` (reasonably good, specific replies
downgraded to generic clarifying questions). *Improvement:* this is close to the
point where a lightweight learned reranker (not full LLM-as-judge) might outperform
further heuristic patching — noted as a real, evidence-backed threshold, not
speculative over-engineering.

## 6. What is misleading about my headline number?

Several things, stated directly rather than buried:

- **"76.1% classifier accuracy" is trained on weak/silver labels, not hand-labeled
  data.** It inherits every systematic error the labeling regex made (documented:
  false positives like "automatic renewal turn off" matching a hardware pattern).
  The true ceiling on a properly hand-labeled training set is unknown.
- **The golden set itself is not human-verified.** Every number that cites it
  (classifier accuracy, retrieval recall, escalation P/R/F1) is only as good as
  Claude's labeling judgment, explicitly flagged `llm_reviewed_pending_human_signoff`
  throughout. This is the single biggest asterisk on every metric in this report.
- **The golden set is deliberately stratified, not representative of natural
  traffic.** Rare intents (Hardware, Vue, Downloads) are 3-8x oversampled relative to
  their true frequency, and Unclear (35-45% of real traffic) is capped at ~6% of the
  golden set. Real-world accuracy on unfiltered traffic would look different —
  probably better on the intents most people actually hit (Account, Purchases), worse
  on the long tail, and dominated by the Unclear-routing behavior that barely shows
  up in this eval set's denominator.
- **"Escalation engine F1=0.55" beats both trivial baselines, but the margin over
  "always escalate" (F1=0.54) is thin**, and both are computed against an unverified
  ground truth. This is presented honestly as "directionally better, not proven
  definitively better."
- **All reply-generation and safety-check numbers were produced in mock/template
  mode**, not with a real LLM. Mock mode is a *stronger* grounding guarantee (it
  literally cannot hallucinate) but a weaker fluency/naturalness one — live-LLM
  numbers would differ in ways not yet measured.
- **Retrieval's "manual" relevance check is Claude-judged on 36 examples**, not
  independently human-validated, and not run at full scale (188).

## 7. What we'd do with one more week
1. Get real human sign-off on the golden set (or at least the 69 `should_escalate=true`
   examples and everything with a `[MANUALLY CORRECTED]` rationale tag) — this
   unblocks trusting every other number.
2. Run the live-LLM reply-generation path and a proper LLM-as-judge pass, validated
   against a human-labeled subset for judge-agreement (the mandatory check this
   project hasn't been able to run yet, for lack of an API key).
3. Address failure mode #1 (Hardware recall) with either augmented training data or
   a lightweight embedding fallback specifically gated to low-confidence cases.
4. Expand the retrieval relevance check from 36 to the full 188 examples, once human
   sign-off exists to make that judgment trustworthy.
5. A/B the IDF-weighting threshold (currently 5.0) against a larger, human-reviewed
   safety-check test set to find the actual precision/recall-optimal setting, rather
   than the current single-anecdote-driven value.
