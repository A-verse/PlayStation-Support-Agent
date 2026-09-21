"""
build_golden_candidates.py — reproducible stratified sampling for the golden
evaluation set. Produces CANDIDATES with rich metadata for human labeling; it
does NOT assign final ground-truth labels (see label_golden_set.py for the
draft-labeling pass, which is explicitly flagged as pending human review).

Sampling design (documented here so the audit trail and the code can't drift
apart):

- Quotas per intent are NOT proportional to natural frequency. Rare-but-real
  intents (Hardware, Vue, Downloads) are deliberately oversampled so the
  golden set can say something statistically meaningful about them at all.
  Unclear/Insufficient Information is capped well below its true ~35-45%
  share -- otherwise the golden set would be dominated by a non-topic and
  starve every real intent of representation.

- Within each intent's quota, sub-quotas pull specific difficulty strata:
    - ambiguous: proxy signal fires for >=2 intents (multi-intent candidates)
    - followup: is_followup=True (context-dependent messages)
    - long: >=30 words (noisier, more real-world Twitter rambling)
    - short: <=4 words (terse messages, common failure mode)
  The remainder is plain random fill from what's left in that intent's pool,
  so "ordinary" cases are still the majority -- this is not a pure
  hard-case set, it's a representative one with deliberate hard-case coverage.

- Everything is seeded (random.seed(GOLDEN_SEED)) so re-running this script
  on the same pairs.jsonl reproduces an identical candidate pool.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

from intent_signals import matched_intents, proxy_intent

GOLDEN_SEED = 20260914  # today's date, arbitrary but fixed and documented

QUOTAS = {
    "Account Access & Recovery": 35,
    "Purchases, Billing & Refunds": 25,
    "Error Codes & Technical Faults": 20,
    "Network & Service Connectivity": 18,
    "Console Troubleshooting (Safe Mode / General Fix)": 18,
    "Downloads & Digital Content": 15,
    "PlayStation Vue / Streaming Service": 15,
    "Hardware / Device Malfunction": 15,
    "Unclear / Insufficient Information": 28,
}

# within each intent's quota, how many slots to reserve for each stratum
# (remainder after these is plain random fill)
SUBQUOTA_FRACTIONS = {"ambiguous": 0.20, "followup": 0.15, "long": 0.10, "short": 0.10}


def load_pairs(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def annotate(pairs):
    for p in pairs:
        text = p["customer_text"]
        p["_matched_intents"] = matched_intents(text)
        p["_proxy_intent"] = proxy_intent(text)
        p["_word_count"] = len(text.split())
        p["_is_ambiguous_signal"] = len(p["_matched_intents"]) >= 2
    return pairs


def sample_for_intent(pool, quota, seed):
    rng = random.Random(seed)
    pool = pool[:]
    rng.shuffle(pool)
    picked = []
    picked_ids = set()

    def take(predicate, n):
        nonlocal picked
        count = 0
        for p in pool:
            if count >= n:
                break
            if p["pair_id"] in picked_ids:
                continue
            if predicate(p):
                picked.append(p)
                picked_ids.add(p["pair_id"])
                count += 1

    n_amb = round(quota * SUBQUOTA_FRACTIONS["ambiguous"])
    n_fol = round(quota * SUBQUOTA_FRACTIONS["followup"])
    n_long = round(quota * SUBQUOTA_FRACTIONS["long"])
    n_short = round(quota * SUBQUOTA_FRACTIONS["short"])

    take(lambda p: p["_is_ambiguous_signal"], n_amb)
    take(lambda p: p["is_followup"], n_fol)
    take(lambda p: p["_word_count"] >= 30, n_long)
    take(lambda p: p["_word_count"] <= 4, n_short)

    remaining_needed = quota - len(picked)
    if remaining_needed > 0:
        take(lambda p: True, remaining_needed)

    return picked[:quota]


def main():
    pairs = annotate(load_pairs("data/processed/pairs.jsonl"))

    by_intent = defaultdict(list)
    for p in pairs:
        by_intent[p["_proxy_intent"]].append(p)

    audit = {"quotas": QUOTAS, "seed": GOLDEN_SEED, "pool_sizes": {}, "sampled_counts": {}, "subquota_fractions": SUBQUOTA_FRACTIONS}
    candidates = []
    for intent, quota in QUOTAS.items():
        pool = by_intent.get(intent, [])
        audit["pool_sizes"][intent] = len(pool)
        picked = sample_for_intent(pool, quota, seed=GOLDEN_SEED)
        audit["sampled_counts"][intent] = len(picked)
        for p in picked:
            candidates.append(p)

    print(f"Total candidates sampled: {len(candidates)}")
    for intent, n in audit["sampled_counts"].items():
        print(f"  {intent:<55} {n:>4}  (pool size {audit['pool_sizes'][intent]:,})")

    # shuffle final order so labeling isn't done intent-block-by-intent-block
    # (reduces anchoring bias during manual review)
    rng = random.Random(GOLDEN_SEED + 1)
    rng.shuffle(candidates)

    out = []
    for i, p in enumerate(candidates):
        out.append(
            {
                "candidate_id": f"cand_{i:04d}",
                "pair_id": p["pair_id"],
                "customer_text": p["customer_text"],
                "prior_context": p["prior_context"],
                "is_followup": p["is_followup"],
                "brand_text_for_reference_only": p["brand_text"],
                "proxy_intent_DO_NOT_USE_AS_GROUND_TRUTH": p["_proxy_intent"],
                "proxy_matched_intents": p["_matched_intents"],
                "word_count": p["_word_count"],
            }
        )

    Path("data/processed/golden_candidates.jsonl").write_text(
        "\n".join(json.dumps(o) for o in out) + "\n"
    )
    Path("data/processed/golden_sampling_audit.json").write_text(json.dumps(audit, indent=2))
    print(f"\nWrote {len(out)} candidates to data/processed/golden_candidates.jsonl")
    print("Wrote sampling audit to data/processed/golden_sampling_audit.json")


if __name__ == "__main__":
    main()
