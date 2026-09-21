# AskPlayStation Support Agent — Hiver SDE Intern Take-Home

A grounded, evaluated AI support agent for **AskPlayStation** (built on the Kaggle
"Customer Support on Twitter" dataset), covering intent classification, retrieval-
grounded reply generation, and an auditable auto-handle/escalate decision engine.

**Core principle followed throughout:** the proof matters more than the system. Every
stage of this project is backed by a real evaluation artifact in `data/processed/`,
and every honesty caveat below is load-bearing — please read them, not just the
headline numbers.

---

## ⚠️ Read this first: golden-set sign-off status

The 189-example golden evaluation set (`data/processed/golden_set.json`) was labeled
by **Claude (LLM-assisted), not an independent human**. Every example's
`_review_status` field says `"llm_reviewed_pending_human_signoff"`. This is disclosed
consistently everywhere it matters — the eval harness, the dashboard, this README —
rather than described as "manually verified." **See `REVIEW_GUIDE.md` for a practical,
prioritized workflow to review and correct it before treating any number here as final.**

---

## Setup (target: <15 minutes)

```bash
git clone <this repo>
cd hiver-support-agent
pip install -r requirements.txt --break-system-packages   # or use a venv
cp .env.example .env   # optional -- see "Enabling the real LLM path" below
```

### Getting the dataset
This project needs the Kaggle **"Customer Support on Twitter"** dataset
(`thoughtvector/customer-support-on-twitter`), specifically `twcs/twcs.csv`
(~493MB, not included in this repo — respect Kaggle's license/terms, download it
yourself):

```bash
# after downloading from Kaggle (requires a Kaggle account + API key):
mkdir -p data/raw
cp /path/to/twcs.csv data/raw/twcs.csv
```

### Reproducing the pipeline from scratch
All of `data/processed/`, `models/`, and the golden set are already included as
build artifacts, so **you do not need to re-run ingestion to try the demo or eval
harness** (skip to "Try it" below). To rebuild everything from raw data:

```bash
python3 src/ingest.py count-brands --raw data/raw/twcs.csv
python3 src/ingest.py build-pairs --raw data/raw/twcs.csv --brand AskPlayStation \
    --max-pairs 100000 --out data/processed/pairs.jsonl
python3 src/build_golden_candidates.py     # NOTE: this reshuffles sampling; re-labeling
python3 src/label_golden_set.py            # 189 examples is substantial manual work --
python3 src/apply_manual_corrections.py    # see the git history / decision log for how
python3 src/apply_manual_corrections_batch2.py  # the shipped golden_set.json was built.
python3 src/apply_policy_decisions.py
python3 src/build_training_set.py
python3 src/train_and_eval_classifier.py
python3 src/build_retrieval_corpus.py
python3 src/retrieval.py
python3 src/eval_retrieval.py
python3 src/eval_harness.py
```

### Try it
```bash
# Interactive demo (recommended first step):
streamlit run app.py

# Or from the command line:
python3 src/agent.py

# Run the test suite:
pytest tests/ -v

# Re-run the full evaluation:
python3 src/eval_harness.py
```

### Enabling the real LLM path
Reply generation works in two modes, and is **not locked to any single provider**:
- **Mock/template mode (default, no key needed):** the reply is built directly from
  the top retrieved historical resolution, with no free-text generation — the
  strongest possible grounding guarantee, at a fluency cost. This is what every
  evaluation number in this repo was produced with.
- **Live LLM mode:** set ANY ONE of these in `.env` (see `.env.example`):
  - `ANTHROPIC_API_KEY` — Claude models
  - `OPENAI_API_KEY` — OpenAI models
  - `LLM_API_KEY` + `LLM_BASE_URL` — any OpenAI-compatible endpoint (Groq, Together,
    Fireworks, Mistral's API, a local Ollama server via its OpenAI-compat shim, etc.)

  The provider is auto-detected (or forced via `LLM_PROVIDER`); `REPLY_GEN_MODEL`
  picks the model. See `src/llm_providers.py` for the detection logic and defaults.
  The UI shows which mode/provider is active — this is never silently swapped.

---

## Architecture

```
Customer message (+ prior context)
        |
        v
Intent classifier (TF-IDF + Logistic Regression, class_weight=balanced)
        |
        v
Message-quality guard  ---->  overrides low-confidence retrieval mode
        |                     even if classifier is confidently wrong
        v
Confidence-gated retrieval (TF-IDF cosine similarity, intent-aware
rerank only when confidence >= 0.6)
        |
        v
Grounded reply generation (mock template OR real LLM, strict grounding
system prompt, evidence always attached)
        |
        v
Response-safety checks (polarity, responsiveness, context-consistency)
  -- fails closed to a safe clarification reply, never silently ships
     a bad candidate
        |
        v
Escalation engine (explicit point-based signal scoring, auditable
reason codes) --> auto_handle (resolve | clarify) | escalate
```

Every stage is its own module (`src/ingest.py`, `train_and_eval_classifier.py`,
`retrieval.py`, `reply_generator.py`, `response_safety.py`, `escalation_engine.py`,
`agent.py`), independently testable and independently evaluated. `agent.py` is the
single composition point both the frontend and eval harness call.

## Why AskPlayStation (brand selection)
Chosen over higher-volume candidates (AmazonHelp, AppleSupport) specifically because
it has the **lowest "please DM us" deflection rate** among viable candidates (25.3%
vs. AppleSupport's 52.5%), meaning its historical replies are disproportionately
*actual troubleshooting content* rather than deflections — the thing "grounded in
historical resolution" depends on. Full comparison in `decision_log.md` entry on
brand selection.

## Intent taxonomy (frozen, 9 labels)
Derived bottom-up from TF-IDF+KMeans clustering + manual reading of the actual data
(not assumed generic PlayStation categories): Account Access & Recovery, Purchases/
Billing & Refunds, Error Codes & Technical Faults, Network & Service Connectivity,
Console Troubleshooting, Downloads & Digital Content, PlayStation Vue/Streaming
Service, Hardware/Device Malfunction, and **Unclear/Insufficient Information** — the
last one deliberately modeled as a *routing state* (safe clarification), not a topic,
since ~35–45% of raw traffic has no identifiable topic at all. See `decision_log.md`
for the full taxonomy derivation and class-imbalance discussion.

---

## Evaluation results (reproduced by `src/eval_harness.py`)

**All numbers below were produced with reply generation in MOCK mode** (no API key
in this environment). Re-run with `ANTHROPIC_API_KEY` set to get live-LLM numbers.

### Intent classification
Trained on 15,806 **weak/silver labels** (a regex heuristic, NOT hand-labeled),
evaluated against the 188-example golden set.

| | Accuracy | Macro F1 |
|---|---|---|
| Trivial baseline (majority class) | 6.4% | 0.013 |
| **TF-IDF + Logistic Regression** | **76.1%** | **0.760** |

*(The majority baseline's low score is a golden-set design feature, not a fluke — the
golden set was deliberately stratified to not be dominated by the majority class. See
`decision_log.md` entry 12.)*

### Retrieval
| | Recall@1 | Recall@3 |
|---|---|---|
| Automatic proxy (same-intent match, n=188) — global | 53.2% | 79.8% |
| Automatic proxy — intent-aware | 67.6% | 77.1% |
| Claude-judged relevance (n=36, **not human-validated**) — global | 47.2% | 77.8% |
| Claude-judged relevance — intent-aware | 55.6% | 77.8% |

### Escalation engine vs. trivial baselines
**Ground truth here is the golden set's `should_escalate` field — itself LLM-reviewed,
pending human sign-off. Treat these numbers as provisional.**

| Approach | Accuracy | Precision | Recall | F1 | False-escalate % | False-auto-handle % |
|---|---|---|---|---|---|---|
| Always auto-handle | 0.63 | 0.00 | 0.00 | 0.00 | 0.0% | 36.7% |
| Always escalate | 0.37 | 0.37 | 1.00 | 0.54 | 63.3% | 0.0% |
| **Escalation engine** | **0.70** | **0.62** | **0.49** | **0.55** | **38.2%** | **26.3%** |

The engine beats both trivial baselines on accuracy and F1, with a materially better
false-escalate rate than "always escalate" and a materially better false-auto-handle
rate than "always auto-handle" — the two failure modes that matter most in opposite
directions (over-escalating wastes human time; under-escalating risks a wrong
confident answer).

### Pipeline-wide (n=188)
- Evidence-failure rate: 39.4%
- Response-safety-failure rate (candidate reply discarded, safe fallback used): 43.1%
- Auto-handle: 70.7% / Escalate: 29.3%

### LLM-as-judge: **not run** (no `ANTHROPIC_API_KEY` in this environment; would mostly
judge template selection in mock mode anyway — re-run once live LLM mode is enabled).

### Human-review agreement: **not available** (no independent human annotations exist
yet — see `REVIEW_GUIDE.md`).

---

## Testing
```bash
pytest tests/ -v
```
12/12 tests pass, covering the escalation engine's 5 required scenarios (normal
resolvable, ambiguous, repeated failed troubleshooting, frustrated-but-resolvable,
safety-check fallback) plus the high-risk override and full audit-trail structure.

## Known limitations (not hidden)
1. **Golden-set labels are LLM-assisted, pending human sign-off** (see above).
2. **Classifier trained on weak/silver labels**, not hand-labeled data — it inherits
   the labeling regex's systematic errors.
3. **Response-safety checks are simple regex/heuristics**, not an LLM reranker — found
   and documented real false positives/negatives during iterative testing (decision
   log entries 15–16); an `idf_overlap_threshold=5.0` parameter is explicitly tunable.
4. **LLM-as-judge and human-agreement evaluation are not run** in this environment
   (no API key, no independent human reviewer yet).
5. **Reply generation defaults to mock/template mode** — fluency-limited by design
   until a real API key is supplied.
6. Full report of top-5 failure modes and "what's misleading about the headline
   number" is in `REPORT.md`.

## Repository structure
```
src/            All pipeline stages (ingestion through escalation engine)
data/processed/ Every evaluation artifact this README cites
models/         Trained classifier + retriever vectorizer
tests/          pytest suite
app.py          Streamlit frontend
decision_log.md 16+ engineering decisions with rationale and trade-offs
REPORT.md       6-page-equivalent report (baselines, failure modes, misleading metrics)
REVIEW_GUIDE.md Practical golden-set human-review workflow
```
