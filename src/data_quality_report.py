"""
data_quality_report.py — produce the data-validation report requested before
moving on to intent taxonomy / model work. Every count either comes straight
from the ingest manifest (single source of truth) or is computed here from
the same final pairs.jsonl, so the report and the pipeline can never drift
apart silently.

Usage:
    python src/data_quality_report.py --pairs data/processed/pairs.jsonl \
        --manifest data/processed/pairs.manifest.json \
        --raw data/raw/twcs.csv --brand AskPlayStation

Note on "topic distribution" and "top recurring issues" below: these are rough
KEYWORD-BASED buckets for reporting purposes only — a quick sanity check that
intent diversity is real and roughly balanced. They are NOT the final intent
taxonomy, which will be built properly (bottom-up, from manual reading) in the
next stage. Don't treat these bucket names as a design decision yet.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

# Deliberately simple, deliberately overlapping-allowed keyword buckets.
# A message can match >1 bucket -- that's fine, this is a rough diagnostic,
# not the real taxonomy (see module docstring).
TOPIC_KEYWORDS = {
    "account_login": [r"\blog\s?in\b", r"\bsign\s?in\b", r"\bpassword\b", r"\baccount\b", r"\bban(ned)?\b", r"\bsuspend", r"psn id"],
    "purchase_billing": [r"\bpurchase", r"\brefund", r"\bcredit card\b", r"\bbill", r"\bcharge", r"\bpayment", r"\bcod points\b", r"\bmoney\b"],
    "error_code_technical": [r"error\s?code", r"\bce-\d", r"\bws-\d", r"\bsu-\d", r"\bnp-\d", r"\bglitch", r"\bbug\b"],
    "hardware": [r"\bcontroller\b", r"\bdisc\b", r"\bconsole\b", r"\boverheat", r"\bhdmi\b", r"turns? off", r"won'?t turn on"],
    "network_server_status": [r"\bserver", r"\bmaintenance\b", r"\bnetwork\b", r"\bconnect", r"\bdown\b", r"\bpsn\b.*down", r"\bwifi\b"],
    "redemption_codes": [r"\bredeem", r"\bcode\b", r"\bvoucher\b", r"\binvalid code\b"],
    "download_digital_content": [r"\bdownload", r"\bdigital\b", r"\blibrary\b", r"\bpre.?order", r"\bgame\b.*(missing|not showing|disappear)"],
}


def load_jsonl(path: Path) -> list:
    with path.open() as f:
        return [json.loads(l) for l in f]


def bucket_topics(pairs: list) -> Counter:
    counts = Counter()
    compiled = {k: [re.compile(p, re.I) for p in pats] for k, pats in TOPIC_KEYWORDS.items()}
    matched_any = 0
    for p in pairs:
        text = p["customer_text"]
        hit = False
        for bucket, patterns in compiled.items():
            if any(pat.search(text) for pat in patterns):
                counts[bucket] += 1
                hit = True
        if hit:
            matched_any += 1
        else:
            counts["unmatched_other"] += 1
    counts["_total_pairs"] = len(pairs)
    counts["_matched_at_least_one_bucket"] = matched_any
    return counts


def top_recurring_issues(raw_pairs: list, top_n: int = 15) -> list:
    """Uses the RAW (pre-dedup) pool so we don't lose frequency signal --
    exact-dedup for the pipeline output removes repeats, but for 'what
    complaints recur most' we want to count them, not collapse them."""
    norm = Counter()
    examples = {}
    for p in raw_pairs:
        key = re.sub(r"\s+", " ", p["customer_text"].lower()).strip()
        norm[key] += 1
        examples.setdefault(key, p["customer_text"])
    return [(examples[k], n) for k, n in norm.most_common(top_n) if n > 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/processed/pairs.jsonl")
    ap.add_argument("--manifest", default="data/processed/pairs.manifest.json")
    ap.add_argument("--raw", default="data/raw/twcs.csv")
    ap.add_argument("--brand", default="AskPlayStation")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    pairs = load_jsonl(Path(args.pairs))

    # Recompute raw (pre-dedup, pre-lang-filter) pairs for frequency analysis.
    # Reuses the exact same reconstruction logic as ingest.py for consistency.
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from ingest import build_pairs_for_brand, load_raw

    df = load_raw(Path(args.raw))
    raw_pairs = build_pairs_for_brand(df, args.brand, context_turns=manifest["context_turns"])

    topic_counts = bucket_topics(pairs)
    recurring = top_recurring_issues(raw_pairs, top_n=15)

    report = {
        "manifest_summary": manifest,
        "rough_topic_bucket_distribution": dict(topic_counts),
        "top_recurring_customer_issues_raw_pool": [
            {"example_text": t, "occurrences": n} for t, n in recurring
        ],
    }

    out_path = Path("data/processed/data_quality_report.json")
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote {out_path}")

    # Human-readable console summary
    m = manifest["counts"]
    print("\n" + "=" * 70)
    print("DATA QUALITY REPORT SUMMARY")
    print("=" * 70)
    print(f"Total paired examples (post-pairing, pre-lang-filter): {m['pairs_after_dedup']:,}")
    print(f"  raw pairs reconstructed:        {m['raw_pairs_reconstructed']:,}")
    print(f"  exact duplicates removed:       {m['exact_duplicates_removed']:,}")
    print(f"English examples (final):        {m['english_pairs']:,}")
    print(f"Non-English examples (dropped):  {m['non_english_pairs']:,}")
    for lang, n in manifest["counts"]["language_distribution"].items():
        if lang != "en":
            print(f"    {lang:<6} {n:>6,}")
    print(f"Near-empty replies (<=4 words):  {m['near_empty_replies_le4_words']:,}")
    print(f"Unique customer tweet IDs:        {manifest['unique_conversations']['unique_customer_tweet_ids']:,}")
    print(f"Unique customer authors:          {manifest['unique_conversations']['unique_customer_author_ids']:,}")
    print(f"Date range:                       {manifest['date_range']['min']}  to  {manifest['date_range']['max']}")
    print("\nRough topic-bucket distribution (keyword-based, diagnostic only):")
    total = topic_counts["_total_pairs"]
    for bucket, n in sorted(topic_counts.items(), key=lambda x: -x[1]):
        if bucket.startswith("_"):
            continue
        print(f"  {bucket:<26} {n:>6,}  ({n/total:.1%})")
    print(f"  {'matched >=1 bucket':<26} {topic_counts['_matched_at_least_one_bucket']:>6,}  ({topic_counts['_matched_at_least_one_bucket']/total:.1%})")
    print("\nTop recurring customer issues (raw pool, exact-text repeats):")
    for t, n in recurring[:10]:
        print(f"  [{n:>3}x] {t[:90]}")


if __name__ == "__main__":
    main()
