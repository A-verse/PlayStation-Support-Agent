"""
ingest.py — turn the raw Kaggle "Customer Support on Twitter" CSV into a clean,
brand-specific set of (customer_message, brand_reply) pairs.

Two modes:

1. Brand survey (run this first, before picking a brand):
     python src/ingest.py count-brands --raw data/raw/twcs.csv

2. Build pairs for the chosen brand:
     python src/ingest.py build-pairs --raw data/raw/twcs.csv --brand AppleSupport \
         --max-pairs 6000 --seed 42 --out data/processed/pairs.jsonl

Design notes (see decision_log.md for the "why"):
- We only use rows where inbound == False and author_id == <brand> as the anchor
  (a real brand reply), then walk in_response_to_tweet_id back to find the
  customer message it answers. This is more reliable than trying to reconstruct
  full multi-turn threads, and it guarantees every pair has a real resolution.
- We optionally attach up to 2 prior customer turns as "context" for cases where
  the immediate reply doesn't make sense standalone (e.g. "yes please" replies).
- Cleaning is deliberately conservative: we do NOT try to fix grammar or spelling
  (that noise is part of the assignment's realism), we only strip boilerplate
  that would leak into retrieval/embedding uselessly (URLs, "DM us" links, etc.)
- Language filtering (langdetect, on by default) keeps only English customer
  messages. This is a real, measured filter (not a manual CSV edit) so it's
  reproducible: rerun this script and you get the same final dataset. Dropped
  non-English pairs are written to a sibling `.non_english.jsonl` file for
  audit, and every count is captured in a `.manifest.json` file that is the
  single source of truth for the data-quality report.
"""

import argparse
import html
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from langdetect import DetectorFactory, LangDetectException, detect
from tqdm import tqdm

# langdetect is non-deterministic by default (it samples n-grams); pin the seed
# so a re-run on the same input produces the same language calls.
DetectorFactory.seed = 0

URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Light, non-destructive cleaning. Keep the noise that matters for realism."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = URL_RE.sub("", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def strip_leading_mentions(text: str) -> str:
    """Remove leading @handle mentions (Twitter puts these at the front of replies),
    but keep any @mentions that appear mid-sentence since those carry meaning."""
    while True:
        stripped = MENTION_RE.sub("", text, count=1).strip()
        if stripped == text or not text.startswith("@"):
            break
        text = stripped
    return text


def load_raw(raw_path: Path) -> pd.DataFrame:
    print(f"Loading {raw_path} ...", file=sys.stderr)
    df = pd.read_csv(
        raw_path,
        dtype={
            "tweet_id": str,
            "author_id": str,
            "response_tweet_id": str,
            "in_response_to_tweet_id": str,
        },
    )
    df["inbound"] = df["inbound"].astype(str).str.lower().isin(["true", "1"])
    print(f"Loaded {len(df):,} rows.", file=sys.stderr)
    return df


def cmd_count_brands(args):
    df = load_raw(Path(args.raw))
    brand_rows = df[~df["inbound"]]
    counts = brand_rows["author_id"].value_counts().head(args.top)
    print("\nTop brand accounts by number of outbound (brand) tweets:\n")
    for brand, n in counts.items():
        print(f"  {brand:<20} {n:>8,}")
    print(
        "\nNext: pick one, skim ~20 of its replies for consistency, then run "
        "`build-pairs --brand <name>`.",
        file=sys.stderr,
    )


def build_pairs_for_brand(df: pd.DataFrame, brand: str, context_turns: int) -> list:
    by_id = df.set_index("tweet_id", drop=False)
    brand_replies = df[(~df["inbound"]) & (df["author_id"] == brand)]
    if len(brand_replies) == 0:
        raise ValueError(f"No rows found for brand '{brand}'. Run count-brands first.")

    pairs = []
    for row in tqdm(brand_replies.itertuples(), total=len(brand_replies), desc="pairing"):
        parent_id = row.in_response_to_tweet_id
        if not isinstance(parent_id, str) or parent_id == "nan" or parent_id not in by_id.index:
            continue
        parent = by_id.loc[parent_id]
        if isinstance(parent, pd.DataFrame):  # duplicate tweet_id edge case
            parent = parent.iloc[0]
        if not parent["inbound"]:
            continue  # brand replying to itself / another brand tweet, skip

        # Walk back up to `context_turns` earlier TURNS (customer or brand,
        # whichever the thread actually alternates through) for extra context.
        # BUG FIX: the original version required every hop to be a customer
        # message and broke the instant it hit a brand reply -- but a normal
        # thread alternates customer -> brand -> customer -> brand, so that
        # check broke on hop 1 almost every time a real conversation existed.
        # Verified against the raw data: 31.8% of pairs are actual
        # mid-conversation follow-ups (their customer tweet directly replies
        # to an earlier brand reply), but the old logic captured context for
        # only ~5% of pairs. Now we keep whichever role the hop actually is
        # and tag it, so downstream consumers (clustering, taxonomy, reply
        # generation) can tell a prior brand reply from a prior customer turn.
        context = []
        cur = parent
        hops = 0
        while hops < context_turns:
            gp_id = cur.get("in_response_to_tweet_id")
            if not isinstance(gp_id, str) or gp_id == "nan" or gp_id not in by_id.index:
                break
            gp = by_id.loc[gp_id]
            if isinstance(gp, pd.DataFrame):
                gp = gp.iloc[0]
            role = "customer" if gp["inbound"] else "brand"
            text = clean_text(gp["text"]) if role == "customer" else strip_leading_mentions(clean_text(gp["text"]))
            context.insert(0, {"role": role, "text": text})
            cur = gp
            hops += 1

        customer_text = clean_text(parent["text"])
        brand_text = strip_leading_mentions(clean_text(row.text))

        if len(customer_text.split()) < 3 or len(brand_text.split()) < 2:
            continue  # too short to be meaningful

        pairs.append(
            {
                "pair_id": f"{brand}_{row.tweet_id}",
                "brand": brand,
                "customer_author_id": parent["author_id"],
                "customer_tweet_id": parent["tweet_id"],
                "customer_text": customer_text,
                "prior_context": context,
                "is_followup": len(context) > 0,
                "brand_tweet_id": row.tweet_id,
                "brand_text": brand_text,
                "created_at": row.created_at,
            }
        )
    return pairs


def dedupe(pairs: list) -> tuple:
    """Exact-match dedup on cleaned customer text. Returns (kept, duplicate_count)."""
    seen = set()
    out = []
    n_dupes = 0
    for p in pairs:
        key = p["customer_text"].lower().strip()
        if key in seen:
            n_dupes += 1
            continue
        seen.add(key)
        out.append(p)
    return out, n_dupes


def detect_lang(text: str) -> str:
    """Best-effort language code for a single message. Returns 'unk' if
    langdetect can't decide (e.g. text too short or no alphabetic content)."""
    try:
        return detect(text)
    except LangDetectException:
        return "unk"


def filter_english(pairs: list) -> tuple:
    """Detect language of each customer message and keep only 'en'.
    Returns (kept_pairs, dropped_pairs, lang_counter) so the caller can report
    the full distribution, not just a pass/fail count."""
    kept, dropped = [], []
    lang_counter = Counter()
    for p in tqdm(pairs, desc="lang-detect"):
        lang = detect_lang(p["customer_text"])
        lang_counter[lang] += 1
        p = {**p, "detected_lang": lang}
        if lang == "en":
            kept.append(p)
        else:
            dropped.append(p)
    return kept, dropped, lang_counter


def cmd_build_pairs(args):
    df = load_raw(Path(args.raw))
    raw_pairs = build_pairs_for_brand(df, args.brand, args.context_turns)
    print(f"Reconstructed {len(raw_pairs):,} raw pairs for brand={args.brand}", file=sys.stderr)

    deduped_pairs, n_dupes = dedupe(raw_pairs)
    print(
        f"{len(deduped_pairs):,} pairs after de-duplication "
        f"({n_dupes:,} exact-duplicate customer messages removed)",
        file=sys.stderr,
    )

    near_empty = [p for p in deduped_pairs if len(p["brand_text"].split()) <= 4]

    if args.lang_filter:
        english_pairs, non_english_pairs, lang_counts = filter_english(deduped_pairs)
        print(
            f"{len(english_pairs):,} pairs kept as English "
            f"({len(non_english_pairs):,} dropped as non-English)",
            file=sys.stderr,
        )
    else:
        english_pairs, non_english_pairs, lang_counts = deduped_pairs, [], Counter()

    random.seed(args.seed)
    random.shuffle(english_pairs)
    final_pairs = english_pairs
    if args.max_pairs and len(final_pairs) > args.max_pairs:
        final_pairs = final_pairs[: args.max_pairs]
    print(f"{len(final_pairs):,} pairs after capping at max-pairs={args.max_pairs}", file=sys.stderr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for p in final_pairs:
            f.write(json.dumps(p) + "\n")
    print(f"Wrote {len(final_pairs):,} pairs to {out_path}", file=sys.stderr)

    if non_english_pairs:
        non_en_path = out_path.with_name(out_path.stem + ".non_english.jsonl")
        with non_en_path.open("w") as f:
            for p in non_english_pairs:
                f.write(json.dumps(p) + "\n")
        print(f"Wrote {len(non_english_pairs):,} dropped non-English pairs to {non_en_path} (audit trail)", file=sys.stderr)

    # Manifest: single source of truth for the data-quality report. Every number
    # in that report should trace back to this file, not be recomputed ad hoc.
    dates = pd.to_datetime(
        [p["created_at"] for p in deduped_pairs], format="%a %b %d %H:%M:%S %z %Y", errors="coerce"
    )
    manifest = {
        "brand": args.brand,
        "raw_csv": str(args.raw),
        "seed": args.seed,
        "context_turns": args.context_turns,
        "lang_filter_applied": args.lang_filter,
        "counts": {
            "raw_pairs_reconstructed": len(raw_pairs),
            "exact_duplicates_removed": n_dupes,
            "pairs_after_dedup": len(deduped_pairs),
            "near_empty_replies_le4_words": len(near_empty),
            "language_distribution": dict(lang_counts.most_common()),
            "english_pairs": len(english_pairs),
            "non_english_pairs": len(non_english_pairs),
            "final_pairs_after_cap": len(final_pairs),
            "max_pairs_cap": args.max_pairs,
        },
        "unique_conversations": {
            "unique_customer_tweet_ids": len(set(p["customer_tweet_id"] for p in deduped_pairs)),
            "unique_customer_author_ids": len(set(p["customer_author_id"] for p in deduped_pairs)),
        },
        "date_range": {
            "min": str(dates.min()) if len(dates) else None,
            "max": str(dates.max()) if len(dates) else None,
        },
        "output_files": {
            "final": str(out_path),
            "non_english_audit": str(out_path.with_name(out_path.stem + ".non_english.jsonl")) if non_english_pairs else None,
        },
    }
    manifest_path = out_path.with_name(out_path.stem + ".manifest.json")
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest to {manifest_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("count-brands", help="Survey brand tweet volume before choosing one")
    p1.add_argument("--raw", required=True)
    p1.add_argument("--top", type=int, default=20)
    p1.set_defaults(func=cmd_count_brands)

    p2 = sub.add_parser("build-pairs", help="Build (customer, brand_reply) pairs for one brand")
    p2.add_argument("--raw", required=True)
    p2.add_argument("--brand", required=True)
    p2.add_argument("--max-pairs", type=int, default=6000)
    p2.add_argument("--context-turns", type=int, default=2)
    p2.add_argument("--seed", type=int, default=42)
    p2.add_argument("--out", default="data/processed/pairs.jsonl")
    p2.add_argument(
        "--lang-filter",
        dest="lang_filter",
        action="store_true",
        default=True,
        help="Keep only English customer messages (default: on)",
    )
    p2.add_argument(
        "--no-lang-filter", dest="lang_filter", action="store_false", help="Disable language filtering"
    )
    p2.set_defaults(func=cmd_build_pairs)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
