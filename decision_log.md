# Decision Log

Each entry: what we chose, why, and alternatives considered.

## 1. Pairing strategy: anchor on brand replies, walk backward
**Chose:** Build (customer_message, brand_reply) pairs by starting from every brand
outbound tweet and following `in_response_to_tweet_id` back to the customer tweet
it answers, rather than trying to reconstruct full conversation trees.
**Why:** Guarantees every pair has a real resolution (the thing we need for
"grounded reply generation"). Full thread reconstruction is more complex and
mostly adds noise for pairs we won't use anyway.
**Alternative considered:** Group by `conversation_id`-style clustering across all
tweet_ids in a thread. Rejected as overkill for this scope — we only need
question → resolution, not the full back-and-forth.

## 2. Context window: up to 2 prior customer turns, not the whole thread
**Chose:** Attach up to 2 earlier customer messages as `prior_context` when
walking back, then stop.
**Why:** Some brand replies ("Sorry to hear that, could you DM us?") only make
sense with a little back-story, but full multi-turn context is rarely needed
and adds prompt-length cost for generation later.
**Alternative considered:** No context at all (simpler, but a nontrivial chunk
of replies are otherwise unintelligible on their own).

## 3. Cleaning is conservative, not corrective
**Chose:** Strip URLs and leading @mentions on the brand side; otherwise leave
text (typos, casing, abbreviations) untouched.
**Why:** The messiness of the language is part of what the assignment wants
handled realistically. Over-cleaning would make the eval look artificially easy.
**Alternative considered:** Full normalization (lowercasing, spelling
correction). Rejected — would inflate retrieval/intent scores in a way that
doesn't reflect real deployment conditions.

## 4. De-duplication key: exact-match on cleaned customer text
**Chose:** Drop pairs whose customer message text (after cleaning) exactly
matches one already kept.
**Why:** Twitter support datasets have a lot of near-identical boilerplate
complaints ("phone won't turn on help!!"); without dedup, the retrieval corpus
and golden set would be dominated by a handful of copy-paste phrasings.
**Alternative considered:** Fuzzy/near-duplicate detection (e.g. MinHash).
Deferred — exact-match after light cleaning removes the worst offenders
cheaply; can revisit if we see near-duplicate pollution in eyeballing.

## 5. Fixed a context-walk bug found during intent-taxonomy EDA
**Found:** Initial TF-IDF+KMeans clustering over customer_text alone produced
several large, semantically empty clusters ("thanks", "I already tried that",
"still doesn't work") — collectively ~47% of pairs. Investigation showed the
`prior_context` field wasn't populating for genuine follow-up turns: the walk-
back logic required every hop to be a customer message and broke immediately
on hitting a brand reply — but real threads alternate customer→brand→customer,
so it broke on hop 1 almost every time. Verified against raw data: 31.8% of
pairs are genuine mid-conversation follow-ups (customer tweet directly replies
to an earlier brand reply), but the buggy logic only captured context for ~5%.
**Chose:** Fixed the walk to follow either role and tag each hop
(`{"role": "customer"|"brand", "text": ...}`), so a follow-up like "still
doesn't work" now carries the actual preceding brand troubleshooting step as
context. Added an explicit `is_followup` flag per pair.
**Why it matters:** Intent taxonomy and later reply-generation both need the
real issue, not just the latest fragment. After the fix, 36.6% of pairs are
follow-ups — a real property of Twitter support threads, not a data bug.
**Alternative considered:** Excluding follow-up turns entirely from the
taxonomy/agent scope (treat only "opening" messages). Rejected: follow-ups
are still real production traffic a deployed agent must handle, and now that
context is captured correctly, they cluster meaningfully rather than as noise.

## 6. Golden-set labeling: rule-based draft + genuine manual review, not silent LLM labeling
**Chose:** Built the golden-set candidate pool via a fully reproducible,
stratified sampling script (fixed seed, documented quotas oversampling rare
intents). Draft labels came from a transparent, inspectable rule-set (regex
patterns encoding the taxonomy definitions and the conservative escalation
criteria) rather than an opaque LLM call. Every escalation-flagged and
multi-intent-flagged example (54), plus 45 random spot-checks across every
intent (99 total, 52% of the set), were then read and manually corrected
where wrong -- 32 genuine corrections were needed, including real regex gaps
(phrasing variants that dodged escalation triggers) and outright mislabels
(e.g. "automatic renewal turn off" false-matching a hardware power-off
pattern).
**Why:** The project explicitly requires a genuinely hand-labeled set with
an accountable human reviewer, not an LLM silently generating ground truth.
Encoding judgment as inspectable rules first, then manually verifying a
majority stratified sample, is more auditable than either pure automation or
an opaque "I read them all" claim.
**Status:** 95/189 (50%) are manually_verified. The remaining 94 are flagged
`rule_based_draft` and need the same manual pass before this is truly final
-- explicitly communicated to the user rather than silently treated as done.
**Alternative considered:** Ask an LLM to generate labels directly. Rejected
per explicit project requirement -- this is exactly what "hand-labeled" is
meant to rule out for the actual assignment submission.

## 7. Retrieval-corpus leakage: flagged for the next stage, not yet resolved
**Found:** The 189 golden-set pair_ids are drawn from the same 15,995-pair
pool that will become the retrieval corpus. If not excluded, the retrieval
step could retrieve a golden example's own historical resolution as "grounding
evidence" for itself, which would make retrieval and reply-quality metrics
look better than they'd be in production.
**Chose (for now):** Just flag it. When the retrieval corpus is built, it
must explicitly exclude all 189 pair_ids in `golden_set_v1.json`.
**Why now, not later:** cheaper to design the retrieval corpus build script
around this exclusion from day one than to discover leakage after the fact.

## 8. Second review batch surfaced a consistent policy: "troubleshooting already exhausted" as an escalation trigger
**Found:** Across the remaining 94 examples, a recurring pattern emerged that
wasn't in the original conservative escalation criteria list: customers who
explicitly state they already tried the standard/documented fix and it failed
("tried all suggested steps, no luck", "did that already, still doesn't
work"). Repeating the same guidance in these cases would be a textbook false
auto-handle -- confidently reprocessing a fix that's already known not to work.
**Chose:** Applied this as an additional escalation trigger consistently
across both review batches (roughly 15 examples), tagged
`troubleshooting_already_exhausted` in `escalation_triggers` so it's
distinguishable from the original criteria (account compromise, bans,
financial disputes, RMA) rather than silently blended in.
**Why it matters for the actual classifier/agent later:** this is a real,
detectable signal (phrases like "tried that", "already did", "didn't work")
that should likely become an explicit feature in the escalation policy, not
just a golden-set labeling artifact.
**Flag for user sign-off:** this is a policy addition beyond what was
explicitly specified -- worth confirming you agree with it as a general rule
before it's built into the agent's escalation logic.

## 9. Real regex/keyword gaps found during full review (not fixed in the rules, only in the manual corrections)
Found multiple cases where the escalation regex missed genuine
escalation-worthy phrasing because of wording variance: "charged...without
authorization" (word order), "a purchase I didn't make" (implies unauthorized
charge without using that word), "gotten took" (informal for
stolen/compromised), "automatic renewal turn off" (false-positive match on a
hardware power-off pattern). These were caught and corrected by hand this
time, but the underlying rule-set was NOT rewritten to catch the general
pattern -- so any twin phrasing elsewhere in the full 15,995-pair corpus is
still not automatically caught. This is a known, documented limitation, not a
silent gap.

## 10. Four examples don't fit the frozen taxonomy and are flagged, not forced
**Found:** cand_0008, cand_0038 (feature/content requests), cand_0123
(a positive customer comment, not a support issue at all), and cand_0183
(a specific how-to question with no clean intent match) don't fit any of the
9 intents' definitions -- not because they're vague (Unclear's actual
definition), but because the taxonomy's scope is "support issues," and these
aren't support issues, or are informational asks the taxonomy wasn't
designed to route.
**Chose:** Did NOT force them into an existing label, and did NOT change the
taxonomy per the explicit instruction to flag rather than silently alter it.
Marked `_review_status: "needs_human_decision"` with the ambiguity spelled
out for each.
**Open question for the user:** whether these represent a real, small
coverage gap worth a policy decision (e.g., "treat as Unclear by convention"
or "exclude non-support chatter from the agent's scope entirely") or whether
they're rare enough (4/189, 2.1%) to not need one.

## 11. User policy decisions on escalation-signal scope and out-of-scope handling
**Escalation:** User approved "troubleshooting already exhausted" as a
signal to feed into the escalation policy, explicitly NOT an absolute rule --
it must be combined with issue severity, confidence, and available evidence,
and customer frustration alone must never automatically trigger escalation.
This will be implemented as one input feature among several when the
escalation engine is built (a later stage), not a standalone gate.
**Taxonomy:** Confirmed frozen, unchanged.
**Out-of-scope examples:** Added explicit `out_of_scope` (bool) and
`out_of_scope_reason` (enum: feature_request, publisher_content_request,
non_support_chatter, out_of_taxonomy_how_to) fields to the golden set schema.
3 of the 4 flagged examples keep a pragmatic `intent="Unclear / Insufficient
Information"` label (for classifier evaluation to still be computable) while
staying identifiable via the new fields. The 4th (pure positive customer
comment, non_support_chatter) is additionally marked
`excluded_from_evaluation=True` and is excluded from classifier accuracy
denominators rather than forced into any label.

## 12. Classifier baseline: TF-IDF + Logistic Regression, trained on weak labels
**Chose:** TF-IDF (unigrams+bigrams, max_features=8000, min_df=2, max_df=0.5)
+ Logistic Regression with `class_weight="balanced"`, trained on 15,806
weakly-labeled examples (regex-heuristic labels from `intent_signals.py`,
the same function used for golden-set stratification -- NOT hand-labeled).
**Why TF-IDF+LogReg over alternatives:** matches the assignment's explicit
guidance (no fine-tuning, no deep learning needed for a baseline). Logistic
Regression over Linear SVM specifically because it outputs calibrated-ish
probabilities, which the later escalation policy will likely want as a
confidence signal (SVM would need extra calibration for that).
**Why weak labels for training, not hand-labeled:** hand-labeling 15,806
examples was never in scope -- only the 189-example golden set is genuinely
hand-reviewed, and it's reserved entirely for evaluation. Training on the
same heuristic used for stratification is transparent and reproducible, but
it means the classifier will inherit the regex's known errors (the ones
found and corrected during golden-set review, e.g. "automatic renewal turn
off" false-matching Hardware). This is a real, documented limitation, not
hidden in the results.
**Context representation:** training and evaluation text is
`prior_context (all turns, either role) + customer_text`, mentions stripped
-- identical enrichment function used during the intent-taxonomy EDA
clustering, so follow-up messages get real topical signal instead of a
context-free fragment.
**Leakage prevention:** `build_training_set.py` explicitly excludes all 189
golden `pair_id`s (loaded from the artifact written during golden-set QA) and
asserts the exclusion count matches exactly. `train_and_eval_classifier.py`
re-asserts disjointness at load time as a second independent check --
leakage would fail loudly, not silently.
**Class imbalance handling:** `class_weight="balanced"` in Logistic
Regression upweights rare classes (Hardware, Vue, Console Troubleshooting)
during training loss, rather than letting the majority class dominate.
Evaluation reports macro F1 (unweighted across classes) alongside weighted F1
and accuracy specifically so imbalance can't hide behind a single headline
number -- and the golden set itself was already deliberately stratified
(not proportional to natural frequency) so rare intents have enough test
examples to produce a meaningful per-class score at all.
**Notable evaluation artifact:** the majority-class trivial baseline scores
a strikingly low 6.4% accuracy on the golden set -- much worse than it would
score on the natural population, precisely because the golden set was
deliberately built to NOT be dominated by the majority class (Unclear).
This needs to be stated explicitly when comparing baselines, or it invites
the wrong conclusion that the majority baseline is "worse than expected" in
some general sense, when really the eval set was built specifically to
resist a majority-class shortcut.

## 13. Retrieval: TF-IDF cosine similarity, in-memory, no vector database
**Chose:** Plain TF-IDF + cosine similarity over the 15,806-document retrieval
corpus (golden-excluded, verified programmatically). No FAISS, no vector DB,
no ANN index.
**Why:** At ~15.8k documents, brute-force cosine similarity over a sparse
TF-IDF matrix runs in milliseconds. A vector database would add real
infrastructure (embedding model, index build/maintenance, serving layer) for
zero measurable benefit at this scale -- exactly the kind of complexity the
assignment explicitly asks us not to add without a demonstrated need. Revisit
only if corpus size or latency requirements change substantially.
**Intent-aware retrieval implemented as a soft rerank, not a hard filter:**
a hard filter (only search within the predicted intent's documents) would
zero out recall whenever the classifier's intent prediction is wrong -- 24%
of the time per the classifier's own eval. Reranking (an additive similarity
bonus for matching intent) is safer: a strongly-matching wrong-intent
document can still win.
**Evaluation methodology, not invented labels:** exact-answer relevance
can't be labeled automatically without fabricating ground truth. Used two
complementary measures instead: (1) an automatic "same-intent" proxy recall
over all 188 golden examples (cheap, coarse, explicitly NOT true relevance),
and (2) genuine relevance judgment on a stratified 36-example subset,
explicitly flagged as Claude-judged pending human spot-check (same caveat
applied throughout this project to any LLM-produced judgment).
**Finding, reported honestly rather than smoothed over:** the automatic
proxy and the manual judgment agree that intent-aware retrieval clearly
helps at k=1, but they disagree in direction at k=3 (proxy shows a small
decrease, manual judgment shows no difference). Reported both rather than
picking the more favorable one.

## 14. Reply generation: confidence-gated retrieval, mock-mode grounding, and a real gap found in testing
**Confidence threshold (0.6) and evidence-failure criteria** are documented
in full in `src/reply_generator.py`'s module docstring (calibration bins,
weak-similarity percentile, conflicting-intent rule) rather than only here,
since that's where an implementer will actually look.
**Mock mode as the strongest grounding guarantee, at a fluency cost:**
when no `ANTHROPIC_API_KEY` is set, the reply is built directly from the
top retrieved historical reply with no free-text generation -- it literally
cannot hallucinate, but it also can't smooth over a retrieval mismatch the
way an LLM might (or might not -- see below). Real-LLM mode uses a strict
grounding system prompt instead, trading some strength-of-guarantee for
fluency and better handling of edge cases (already-tried context, tone).
**Real limitation found during the smoke test, not swept under the rug:**
`cand_0047` ("Ps4 Account") was classified as Account Access & Recovery with
92% confidence -- high enough to pass the 0.6 gate confidently -- but the
golden-set's true label is Unclear/Insufficient Information (the message has
no actual content). The evidence-quality check didn't catch this because it
only flags `unclear_intent` when the *predicted* intent is Unclear, not when
the true intent is Unclear but the classifier is confidently wrong. The
resulting reply ("check out the next article to report this user") is
unhelpful and unrelated. This is a real gap: overconfident-and-wrong is not
the same failure mode as low-confidence, and the current evidence-quality
check only catches the second one. Flagged for the escalation-engine stage
rather than patched here, since a proper fix likely needs a signal beyond
what retrieval/classification alone can provide (e.g. message length as an
input, not just classifier confidence).

## 15. Response-safety safeguards: two real bugs found and fixed before finalizing, one left open and documented
**Bug 1 (fixed): curly-apostrophe blindness.** All safety regexes (`won't`,
`doesn't`, etc.) only matched straight ASCII apostrophes, but real customer
text (and this corpus) predominantly uses curly quotes (`won\u2019t`). This
caused widespread false positives in the message-quality guard
(`no_actionable_request` firing on messages that clearly did report a
problem). Fixed with a single `normalize()` step applied before every regex
check in `response_safety.py`, rather than patching each pattern individually.
**Bug 2 (fixed): responsiveness check compared against the wrong text.**
The topic-overlap check originally compared the query against the retrieved
document's bare `customer_text`, but retrieval itself matched on the
enriched text (context + customer_text). This caused false "topic mismatch"
flags whenever a retrieved document's relevance came from ITS OWN prior
context. Fixed by folding the retrieved document's `prior_context` into the
comparison, matching what retrieval actually indexed on.
**Limitation found and left open (documented, not silently fixed):** the
query side of the same overlap check still only uses the CURRENT turn's
text, not the query's own prior_context. This causes real false positives
on terse follow-ups (`cand_0105`, `cand_0039`) where the actual topic lives
in context the customer isn't repeating. A second, related limitation: the
overlap check does exact word matching with no stemming, so "install" vs
"installing" or "download" vs "downloading" register as zero overlap despite
being the same topic (`cand_0150`). Both are documented here rather than
patched immediately, per the instruction to keep safeguards simple and only
add complexity (e.g. stemming, or folding context into the query side too)
if the simple version demonstrably falls short at scale -- these two
examples are evidence toward that threshold, not proof of it yet.

## 16. Context-aware overlap check: two more iterations to get right, final IDF-weighted approach
**Iteration 1 (query-side context, unconditional):** folding prior_context
into the query side of the overlap check unconditionally fixed the intended
false positives (`cand_0105`, `cand_0039`) but introduced a real regression:
`cand_0035` (one of the original 5 known-bad examples) stopped being caught,
because its own prior context happened to mention "safe mode / restore
default settings" -- generic boilerplate that overlapped with the retrieved
doc's unrelated context and masked a genuine topic mismatch.
**Iteration 2 (gate context supplementation on a backward-reference
phrase):** only fold in context when the current turn contains an explicit
backward reference ("I did that", "my original tweet", "read the entire
thing") rather than doing it unconditionally or by message length. This
correctly stopped contaminating cand_0035 (no backward reference in its own
text) while still fixing cand_0105/cand_0039. But cand_0035 was STILL not
caught -- the contamination was coming from the RETRIEVED DOC's own context
(added in an earlier fix), not the query side.
**Final fix: IDF-weighted overlap, not a flat set intersection.** "safe" and
"mode" are near-universal across Console Troubleshooting conversations
(IDF ~4.2-4.4 in the fitted corpus), so any two such conversations will
share them regardless of whether the specific sub-issue matches. Requiring
at least one shared word with IDF >= 5.0 (reusing the retriever's own
already-fitted TF-IDF vocabulary/idf_ -- no new model or dependency) means
common boilerplate can't manufacture false "relevance" on its own; a
genuinely specific/rare shared term is required. This fixed cand_0035
without breaking cand_0105/cand_0039/cand_0150.
**Trade-off found and reported, not hidden:** with this stricter check,
total changed replies in the 45-example smoke test rose from 12 to 18. Most
of the additional 6 are a mild QUALITY cost, not a safety one -- a
reasonably specific clarifying question (e.g. "what were you doing when you
got this error?") gets replaced by the generic fallback clarifying question,
never by something wrong or harmful. `idf_overlap_threshold=5.0` is an
explicit, documented, tunable parameter; lowering it would reduce this
quality cost at the risk of re-admitting cases like cand_0035. Left at 5.0
per the explicit instruction not to weaken the safeguard to reduce fallback
count.

## 17. Escalation engine: explicit additive point system, not a learned model
**Chose:** A documented point system (HIGH_RISK=+3, most other signals=+1,
FRUSTRATION=+0 always) with a fixed threshold (2), fully inspectable per
decision via `signal_breakdown` and `decision_inputs`.
**Why:** Matches the project's explainability requirement -- any decision
can be explained in one sentence ("escalated because high-risk intent
detected" or "auto-handled because only one weak signal fired, below
threshold"). A learned model would need its own training data and
validation, which isn't available yet, and would be far harder to defend
in an interview than an explicit, auditable rule.
**Key design choice validated by the approved policy:** FRUSTRATION_DETECTED
contributes 0 points ALWAYS -- logged for observability, never a deciding
factor, directly implementing "frustration alone must not trigger
escalation." EXHAUSTED_TROUBLESHOOTING_SIGNAL contributes only +1 (a signal,
not a rule) -- needs to combine with at least one other signal to escalate,
implementing "must be combined with severity, confidence, evidence."
**Alternative considered:** A decision tree with hand-written branches per
combination. Rejected -- an additive score is simpler to reason about and
extend (new signals just need a point value) than an exponentially-growing
branch structure.

## 18. Evaluation harness: strict labeling discipline over convenient rounding-up
**Chose:** Every metric in `eval_harness.py`'s output is explicitly labeled
as one of: automatic proxy, Claude-judged (not independent human), or
"not run/not available" (LLM-as-judge, human-review agreement) -- never
silently upgraded to sound more rigorous than it is.
**Why:** The project's own instructions were explicit that mislabeling
LLM-assisted work as "manually verified" is a real error to avoid, not a
minor wording nit. Given the golden set's `_review_status` already tracks
this per-example, the harness just has to not throw that information away
when aggregating.
**Escalation baselines chosen:** "always auto-handle" and "always
escalate" -- the same trivial-baseline pattern used for intent
classification (majority-class baseline), applied consistently to the
escalation decision. Both trivial baselines are computed against the same
(caveated) ground truth as the engine itself, so the comparison is
apples-to-apples.

## 19. Frontend: Streamlit over a custom web stack
**Chose:** Streamlit, connected directly to the real `Agent` class (no
mock data, no separate API layer) -- the whole app is ~150 lines because it
imports the pipeline modules directly rather than serializing over HTTP to
a separate backend.
**Why:** The existing stack is pure Python; adding a JS framework and a
separate API server would be new infrastructure for a demo tool, which the
assignment explicitly asks to avoid unless something simpler doesn't work.
Streamlit's direct-import model also guarantees the UI can't drift from the
actual pipeline behavior, since there's no serialization boundary to get
out of sync.
**Mock mode is shown explicitly in the UI** (a banner stating which mode is
active), not hidden -- consistent with the project-wide rule that mock
mode must be labeled, never silently presented as equivalent to live LLM
output.

## 20. Multi-provider LLM support: thin abstraction over Anthropic/OpenAI/OpenAI-compatible
**Chose:** A single `llm_providers.py` module with one `get_provider()` factory
and one `.complete(system, user, max_tokens)` method, auto-detected from
whichever API key env var is set (`ANTHROPIC_API_KEY` > `OPENAI_API_KEY` >
`LLM_API_KEY`+`LLM_BASE_URL`), with an explicit `LLM_PROVIDER` override.
**Whyःnot locking to one vendor:** the OpenAI-compatible path (generic
base_url) covers Groq, Together, Fireworks, Mistral's API, and local Ollama
in one code path, since they all speak the same chat-completions wire
format -- no per-vendor SDK needed beyond the `openai` package itself.
**Why this design over a heavier abstraction (e.g. LangChain):** the actual
requirement is one method with one signature; a full framework would add
a dependency and an abstraction layer for something `reply_generator.py`
already called in exactly one place. Kept it to ~120 lines, fully covered
by 8 fast unit tests that check detection/error-handling logic without
needing real API credentials.
**Tested without real keys:** provider *detection* and *error-handling*
(e.g. `openai_compatible` without `LLM_BASE_URL` raises a clear
`ValueError`) are fully unit-tested; actual live completions are
necessarily untested in this environment (no credentials available) and
are called out as such in README/REPORT rather than assumed to work.

## 21. UI polish: refined within Streamlit's native responsive behavior, not custom breakpoints
**Chose:** Custom CSS (accent palette, card styling, status pills) layered
on top of Streamlit's built-in column auto-stacking (native since 1.28+),
plus one added media query for sub-640px typography/padding, rather than
reimplementing responsive layout from scratch.
**Why:** Streamlit already collapses `st.columns` into a vertical stack on
narrow viewports; fighting that with a custom grid system would be the kind
of unnecessary infrastructure the project has avoided elsewhere. The added
CSS is refinement (spacing, color, type scale), not a replacement for the
framework's own responsive behavior.
**Added a third "What's Next" tab** reading directly from `REPORT.md`'s
"one more week" section (not a separately maintained copy), so the roadmap
shown in the UI can't drift out of sync with the written report.
