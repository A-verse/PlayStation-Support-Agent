# Golden-Set Human Review Guide

**Status: the 189-example golden set (`data/processed/golden_set.json`) has NOT been
human-verified.** Every example's `_review_status` field is one of:

- `"llm_reviewed_pending_human_signoff"` (185 examples) — Claude read the example
  against the taxonomy definitions and escalation criteria and assigned/corrected the
  label, with a written rationale. This is a considered draft, not ground truth.
- `"needs_human_decision"` (originally 4; now resolved per your explicit policy
  decisions — see `decision_log.md` entries 10–11 — but still worth a second look)

**Do not cite this golden set's numbers as "human-validated" until you've actually
reviewed it.** The evaluation harness (`src/eval_harness.py`) and README both carry
this caveat consistently rather than upgrading the label quietly.

## How to review it efficiently

You don't need to re-read all 189 from scratch — the fields are built for a fast
scan-and-correct pass, not a blind re-label.

1. **Open `data/processed/golden_set.json`.** Each example has:
   - `customer_text`, `prior_context` — what you're judging
   - `intent`, `multi_intent_candidates` — Claude's judgment
   - `should_escalate`, `escalation_triggers` — Claude's judgment + why
   - `label_rationale` — the actual reasoning, including any correction history
     (look for `[MANUALLY CORRECTED]`, `[RESOLVED PER USER POLICY]`, etc. prefixes —
     these mark places where the label changed during review, worth a second look)
   - `out_of_scope`, `out_of_scope_reason`, `excluded_from_evaluation` — the 4
     out-of-taxonomy cases

2. **Prioritize by risk, not alphabetically:**
   - First: every example where `should_escalate=true` (69 examples) — these drive
     the escalation-engine's precision/recall numbers directly, so errors here are
     the most consequential.
   - Second: every example whose `label_rationale` contains `[MANUALLY CORRECTED]`
     (see `decision_log.md` entries 6–10 for what changed and why) — these are
     where Claude's initial rule-based read was wrong and got fixed; worth
     independently confirming the fix was actually right.
   - Third: a random spot-check of the rest for general confidence.

3. **To correct a label:** edit the JSON directly (it's just a list of dicts) or
   write a small script following the pattern in `src/apply_manual_corrections.py`
   (keeps a change log rather than silently overwriting). Whichever you use, please
   update `_review_status` to `"human_verified"` for anything you've actually
   checked, so the eval harness (which currently reports everything as
   LLM-reviewed) can be updated to reflect real human sign-off counts.

4. **Re-run after corrections:**
   ```bash
   python3 src/qa_golden_set_final.py   # re-validates schema, duplicates, counts
   python3 src/eval_harness.py          # re-runs full pipeline eval against corrected labels
   ```

## What changes once you've signed off

Nothing in the pipeline code needs to change. The only things that change are:
- `_review_status` values in `golden_set.json`
- The eval harness and README's caveat language (from "pending sign-off" to
  "human-verified"), which should be updated to match reality once you've done it —
  don't just delete the caveat without actually completing the review.
