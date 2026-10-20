#!/usr/bin/env python3
"""Build and freeze semcache/pairs/pairs.jsonl: trap pairs (label "different") and control pairs ("same").

Sources, all frozen in the repo:
  template      the TRAPS / CONTROLS tables below, expanded and sampled with SEED
  handwritten   semcache/pairs/handwritten.jsonl, written by Claude Opus 5.5 (claude-opus-5-5) in the authoring
                session on 2026-10-19 - the "LLM" half of "templates plus an LLM". No API call made them, so
                they can't be regenerated; they're committed as-is.
  generated     semcache/pairs/generated.jsonl from semcache/generate_pairs.py: gpt-6.1-sol and Claude Sonnet 5.5
                writing from the slot-type spec alone, plus BANKING77 real queries they edit (ids g000+).

Then: drop exact duplicates (case/space/punctuation-insensitive) and near-duplicates (rapidfuzz token_set_ratio
>= 95 on both sides against a kept pair of the same label), keep at most PER_SLOT_TRAPS traps per slot type
(handwritten first, then templates in seeded order), write pairs.jsonl, a 30-row review CSV, and stats.

pairs.jsonl is frozen: the script refuses to overwrite it unless --refreeze, and MANIFEST.json records its sha256.

    python -m semcache.build_pairs            # build (first time) or verify the frozen file matches a rebuild
    python -m semcache.build_pairs --refreeze
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict

from rapidfuzz import fuzz

from .common import HANDWRITTEN, N_REVIEW, PAIRS, PAIRS_DIR, SEED

GENERATED = PAIRS_DIR / "generated.jsonl"

SLOT_TYPES = ("class", "direction", "negation", "version", "date_time", "number", "entity")
PER_SLOT_TRAPS = 29          # 7 x 29 = 203
TEMPLATE_TRAPS_PER_SLOT = 16
N_CONTROLS = 200
TEMPLATE_CONTROLS = 70
NEAR_DUP = 95

# ------------------------------------------------------------------------------------------- trap templates
# (slot_type, domain, template, [(value_a, value_b), ...]); a template with {a} and {b} gets a pair of tuples.
TRAPS: list[tuple[str, str, str, list]] = [
    # cabin class
    ("class", "airline", "What is the checked baggage allowance in {x}?",
     [("Economy", "Premium Economy"), ("Premium Economy", "Business"), ("Business", "First Class")]),
    ("class", "airline", "Is lounge access included with an Emirates {x} ticket?",
     [("Economy", "Premium Economy"), ("Premium Economy", "Business Class"), ("Business Class", "First Class")]),
    ("class", "airline", "How much does it cost to change my {x} fare from DXB to LHR?",
     [("Economy", "Premium Economy"), ("Economy", "Business"), ("Business", "First Class")]),
    ("class", "airline", "What meals are served in {x} on flights over 8 hours?",
     [("Economy", "Premium Economy"), ("Premium Economy", "Business")]),
    ("class", "airline", "How many Skywards miles do I earn flying {x} to Sydney?",
     [("Economy", "Premium Economy"), ("Business", "First Class")]),
    ("class", "airline", "What is the seat pitch in {x} on the A350?",
     [("Economy", "Premium Economy"), ("Premium Economy", "Business")]),
    # direction
    ("direction", "airline", "Cheapest flights from {a} to {b} next month",
     [(("DXB", "LHR"), ("LHR", "DXB")), (("Singapore", "Sydney"), ("Sydney", "Singapore")),
      (("JFK", "CDG"), ("CDG", "JFK"))]),
    ("direction", "airline", "Is there a direct flight from {a} to {b} on weekdays?",
     [(("Doha", "Mumbai"), ("Mumbai", "Doha")), (("FRA", "BKK"), ("BKK", "FRA"))]),
    ("direction", "banking", "What is today's exchange rate from {a} to {b}?",
     [(("USD", "EUR"), ("EUR", "USD")), (("GBP", "AED"), ("AED", "GBP")), (("INR", "USD"), ("USD", "INR"))]),
    ("direction", "banking", "How long does a transfer from my {a} account to my {b} account take?",
     [(("savings", "checking"), ("checking", "savings"))]),
    ("direction", "dev-tools", "What's the best way to migrate a production database from {a} to {b}?",
     [(("MySQL", "PostgreSQL"), ("PostgreSQL", "MySQL")), (("MongoDB", "Postgres"), ("Postgres", "MongoDB"))]),
    ("direction", "dev-tools", "How do I convert a codebase from {a} to {b}?",
     [(("JavaScript", "TypeScript"), ("TypeScript", "JavaScript")), (("Java", "Kotlin"), ("Kotlin", "Java"))]),
    ("direction", "support", "Can I switch from the {a} plan to the {b} plan in the middle of a billing cycle?",
     [(("monthly", "annual"), ("annual", "monthly"))]),
    # negation
    ("negation", "airline", "What happens if I {x} check in online before my flight?", [("", "don't")]),
    ("negation", "airline", "Which items {x} allowed in my carry-on bag?", [("are", "are not")]),
    ("negation", "banking", "What happens if I {x} pay the minimum amount by the due date?", [("", "don't")]),
    ("negation", "banking", "Why {x} my card working at foreign ATMs?", [("is", "isn't")]),
    ("negation", "dev-tools", "Why {x} pip install the package into my virtualenv?", [("does", "doesn't")]),
    ("negation", "dev-tools", "What happens if I {x} pin my dependency versions?", [("", "don't")]),
    ("negation", "support", "What happens if I {x} renew my subscription before it expires?", [("", "don't")]),
    ("negation", "support", "Which features {x} included in the free plan?", [("are", "are not")]),
    ("negation", "support", "Why {x} receiving the verification email?", [("am I", "am I not")]),
    # version
    ("version", "dev-tools", "What's new in {x}?",
     [("Python 3.11", "Python 3.12"), ("Python 3.12", "Python 3.13"), ("Node 20", "Node 22"), ("Django 4.2", "Django 5.0")]),
    ("version", "dev-tools", "When does {x} reach end of life?",
     [("Python 3.11", "Python 3.12"), ("Node 18", "Node 20"), ("PostgreSQL 15", "PostgreSQL 16")]),
    ("version", "dev-tools", "What are the breaking changes when upgrading to {x}?",
     [("Java 17", "Java 21"), ("React 18", "React 19"), ("Kubernetes 1.29", "Kubernetes 1.30")]),
    ("version", "support", "What changed in app version {x}?", [("5.2", "5.3"), ("4.10", "4.11")]),
    ("version", "dev-tools", "What is the default rate limit on API {x}?", [("v1", "v2")]),
    # date or time
    ("date_time", "airline", "What's the status of flight EK 1 on {x}?", [("3 March", "4 March"), ("today", "tomorrow")]),
    ("date_time", "airline", "Which flights leave DXB for LHR after {x}?", [("9am", "3pm"), ("noon", "6pm")]),
    ("date_time", "banking", "What was the USD to EUR exchange rate on {x}?", [("1 June", "1 July"), ("March 31", "April 30")]),
    ("date_time", "banking", "Show me all card transactions from {x}", [("Monday", "Tuesday"), ("last week", "this week")]),
    ("date_time", "dev-tools", "Which deploys went out on {x}?", [("2026-10-01", "2026-10-02"), ("Friday", "Saturday")]),
    ("date_time", "support", "Is phone support open at {x} on Sunday?", [("8am", "8pm"), ("7:30", "21:30")]),
    ("date_time", "airline", "What is the gate for my {x} flight to Doha?", [("9:15am", "9:15pm")]),
    # number or amount
    ("number", "banking", "Do I need extra verification to send {x} abroad?", [("$500", "$50,000"), ("€2,000", "€20,000")]),
    ("number", "banking", "What's the monthly payment on a {x} loan over 5 years?", [("$10,000", "$25,000"), ("$100k", "$200k")]),
    ("number", "banking", "How much is {x} in euros?", [("100 USD", "1,000 USD"), ("$250", "$2,500")]),
    ("number", "airline", "How much is the excess baggage charge for a {x} bag?", [("23 kg", "32 kg"), ("25 kg", "30 kg")]),
    ("number", "airline", "Can I bring a {x} bottle of liquid in my carry-on?", [("100 ml", "500 ml")]),
    ("number", "support", "What's the monthly price for {x} on the Team plan?", [("5 users", "50 users"), ("10 users", "100 users")]),
    ("number", "dev-tools", "How long does it take to upload a {x} file to S3?", [("1 GB", "100 GB")]),
    ("number", "support", "What do I get after {x} days of the free trial?", [("7", "30")]),
    # entity swap
    ("entity", "banking", "What's the customer service number for {x}?", [("Visa", "Mastercard"), ("Chase", "Citi"), ("HSBC", "Barclays")]),
    ("entity", "banking", "Does {x} charge a foreign transaction fee on my card?", [("Visa", "American Express"), ("Mastercard", "Amex")]),
    ("entity", "airline", "What's the cabin baggage size limit on {x}?", [("Emirates", "Qatar Airways"), ("Ryanair", "easyJet"), ("Lufthansa", "Air France")]),
    ("entity", "dev-tools", "How do I deploy a Docker container to {x}?", [("AWS", "GCP"), ("Azure", "Google Cloud")]),
    ("entity", "dev-tools", "How do I add an SSH key to my {x} account?", [("GitHub", "GitLab"), ("GitLab", "Bitbucket")]),
    ("entity", "support", "How do I clear the cache in {x}?", [("Chrome", "Firefox"), ("Safari", "Chrome")]),
    ("entity", "support", "How do I install the app on {x}?", [("iOS", "Android"), ("Windows", "macOS")]),
    ("entity", "support", "How do I contact support through {x}?", [("WhatsApp", "email"), ("live chat", "Slack")]),
]

# ------------------------------------------------------------------------------------------- control templates
# (alias_type, domain, template, [(surface_a, surface_b), ...]): same meaning, different surface form.
CONTROLS: list[tuple[str, str, str, list]] = [
    ("direction", "airline", "Cheapest flights from {a} to {b} next month",
     [(("Dubai", "London"), ("DXB", "LHR")), (("Paris", "New York"), ("CDG", "JFK")), (("Singapore", "Sydney"), ("SIN", "SYD"))]),
    ("direction", "airline", "Is there a direct flight from {a} to {b}?",
     [(("Doha", "Mumbai"), ("DOH", "BOM")), (("Frankfurt", "Bangkok"), ("FRA", "BKK")), (("Heathrow", "Dubai"), ("LHR", "DXB"))]),
    ("direction", "banking", "What's today's exchange rate from {a} to {b}?",
     [(("US dollars", "euros"), ("USD", "EUR")), (("pounds", "dirhams"), ("GBP", "AED"))]),
    ("class", "airline", "What is the checked baggage allowance in {x}?",
     [("business class", "biz class"), ("Premium Economy", "premium eco"), ("Economy", "economy class")]),
    ("class", "airline", "Can I upgrade to {x} with miles?", [("business class", "biz"), ("First Class", "first class")]),
    ("class", "airline", "Is wifi free in {x}?", [("Business", "business class"), ("Economy", "coach")]),
    ("version", "dev-tools", "Is pandas 2.2 compatible with {x}?", [("3.12", "Python 3.12"), ("Python 3.11", "py 3.11")]),
    ("version", "dev-tools", "How do I install {x} on Ubuntu?", [("Node 20", "Node.js 20"), ("Python 3.13", "python3.13")]),
    ("version", "dev-tools", "What's the rate limit on {x}?", [("API v2", "version 2 of the API")]),
    ("version", "support", "What changed in {x}?", [("app version 5.2", "v5.2 of the app")]),
    ("number", "banking", "Do I need extra verification to send {x} abroad?",
     [("$1,000", "1000 dollars"), ("$5k", "$5,000"), ("€2,000", "2000 euros")]),
    ("number", "airline", "How much is the excess baggage charge for a {x} bag?", [("23 kg", "23kg"), ("32 kg", "32 kilos")]),
    ("number", "support", "What's the price for {x} on the Team plan?", [("10 users", "ten users")]),
    ("date_time", "airline", "What's the status of flight EK 1 on {x}?", [("3 March", "March 3rd"), ("4 Nov", "November 4")]),
    ("date_time", "airline", "Which flights leave DXB for LHR after {x}?", [("3pm", "15:00"), ("9am", "9:00 am")]),
    ("date_time", "banking", "Show me all card transactions from {x}", [("1 June", "June 1st")]),
    ("entity", "banking", "What's the foreign transaction fee on my {x} card?", [("Amex", "American Express"), ("Mastercard", "MC")]),
    ("entity", "dev-tools", "How do I deploy a container to {x}?", [("GCP", "Google Cloud"), ("AWS", "Amazon Web Services")]),
    ("entity", "dev-tools", "How do I restart a pod in {x}?", [("Kubernetes", "k8s")]),
    ("entity", "airline", "What's the cabin baggage size limit on {x}?", [("British Airways", "BA")]),
    ("negation", "airline", "Why {x} I check in online?", [("can't", "cannot")]),
    ("negation", "banking", "What happens if I {x} pay the minimum amount by the due date?", [("don't", "do not")]),
    ("negation", "dev-tools", "Why {x} pip install the package?", [("doesn't", "does not")]),
    ("negation", "support", "Why {x} receiving the verification email?", [("am I not", "aren't I")]),
]


def _fill(template: str, v) -> str:
    if isinstance(v, tuple):
        return template.format(a=v[0], b=v[1])
    return re.sub(r"\s+", " ", template.format(x=v)).strip()


def expand(table, label: str) -> list[dict]:
    out = []
    for slot, domain, tpl, values in table:
        for va, vb in values:
            out.append({"cached": _fill(tpl, va), "new": _fill(tpl, vb), "label": label, "slot_type": slot,
                        "domain": domain, "source": "template"})
    return out


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9$€£]+", " ", text.lower()).strip()


def dedupe(rows: list[dict]) -> tuple[list[dict], Counter]:
    kept: list[dict] = []
    dropped: Counter = Counter()
    seen: set[tuple[str, str]] = set()
    for r in rows:
        k = (_key(r["cached"]), _key(r["new"]))
        if k[0] == k[1]:
            dropped["identical_sides"] += 1
            continue
        if k in seen or k[::-1] in seen:
            dropped["exact_duplicate"] += 1
            continue
        if any(o["label"] == r["label"]
               and fuzz.token_set_ratio(_key(o["cached"]), k[0]) >= NEAR_DUP
               and fuzz.token_set_ratio(_key(o["new"]), k[1]) >= NEAR_DUP for o in kept):
            dropped["near_duplicate"] += 1
            continue
        seen.add(k)
        kept.append(r)
    return kept, dropped


def _authored() -> tuple[list[dict], Counter, dict]:
    rng = random.Random(SEED)
    hand = [json.loads(l) for l in HANDWRITTEN.open()] if HANDWRITTEN.exists() else []
    for h in hand:
        h["source"] = "handwritten"
    t_traps, t_controls = expand(TRAPS, "different"), expand(CONTROLS, "same")
    rng.shuffle(t_traps)
    rng.shuffle(t_controls)

    # handwritten first within each slot type, then templates, capped
    traps: list[dict] = []
    for s in SLOT_TYPES:
        tt = [r for r in t_traps if r["slot_type"] == s][:TEMPLATE_TRAPS_PER_SLOT]
        hh = [r for r in hand if r["label"] == "different" and r["slot_type"] == s]
        kept, _ = dedupe(hh + tt)
        traps += kept[:PER_SLOT_TRAPS]
    hc = [r for r in hand if r["label"] == "same"]
    controls, _ = dedupe(hc + t_controls[:TEMPLATE_CONTROLS])
    controls = controls[:N_CONTROLS]

    rows, dropped = dedupe(traps + controls)  # final pass across everything
    return rows, dropped, {"template_traps": len(t_traps), "template_controls": len(t_controls),
                           "handwritten": len(hand)}


def authored_rows() -> list[dict]:
    """Templates + handwritten (ids p000+), before ids are assigned. generate_pairs dedupes against these."""
    return _authored()[0]


def source_group(r: dict) -> str:
    """authored | gpt-6.1-sol | claude-sonnet-5-5 (BANKING77-seeded pairs count with their generator)."""
    return r["source"].split("llm:")[1] if "llm:" in r["source"] else "authored"


def build() -> tuple[list[dict], dict]:
    authored, dropped, candidates = _authored()
    for i, r in enumerate(authored):
        r["id"] = f"p{i:03d}"
    generated = [json.loads(l) for l in GENERATED.open()] if GENERATED.exists() else []
    for i, r in enumerate(generated):
        r["id"] = f"g{i:03d}"
    rows = [{k: r[k] for k in ("id", "cached", "new", "label", "slot_type", "domain", "source")}
            for r in authored + generated]
    stats = {
        "n": len(rows), "by_label": Counter(r["label"] for r in rows),
        "traps_by_slot": Counter(r["slot_type"] for r in rows if r["label"] == "different"),
        "controls_by_slot": Counter(r["slot_type"] for r in rows if r["label"] == "same"),
        "by_domain_label": Counter(f"{r['domain']}|{r['label']}" for r in rows),
        "by_source_label": Counter(f"{r['source']}|{r['label']}" for r in rows),
        "by_group_label": Counter(f"{source_group(r)}|{r['label']}" for r in rows),
        "candidates": {**candidates, "generated_kept": len(generated)},
        "dropped_final_pass": dropped,
    }
    return rows, stats


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_review(rows: list[dict]) -> None:
    """N_REVIEW rows, split evenly across source groups (authored, each generator model)."""
    rng = random.Random(SEED + 1)
    groups = sorted({source_group(r) for r in rows})
    pick = []
    for k, g in enumerate(groups):
        n = N_REVIEW // len(groups) + (1 if k < N_REVIEW % len(groups) else 0)
        pick += rng.sample([r for r in rows if source_group(r) == g], n)
    with (PAIRS_DIR / "review.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label", "slot_type", "domain", "source", "cached_query", "new_query",
                    "label_ok (y/n)", "notes"])
        for r in sorted(pick, key=lambda r: r["id"]):
            w.writerow([r["id"], r["label"], r["slot_type"], r["domain"], r["source"], r["cached"], r["new"], "", ""])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refreeze", action="store_true")
    args = ap.parse_args()
    rows, stats = build()
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    if PAIRS.exists() and not args.refreeze:
        same = PAIRS.read_text() == text
        print(f"{PAIRS} is frozen; a rebuild {'matches it byte for byte' if same else 'DIFFERS - not overwritten'}")
        return
    PAIRS_DIR.mkdir(parents=True, exist_ok=True)
    PAIRS.write_text(text)
    write_review(rows)
    (PAIRS_DIR / "stats.json").write_text(json.dumps(stats, indent=1, default=dict))
    (PAIRS_DIR / "MANIFEST.json").write_text(json.dumps(
        {"pairs.jsonl": _sha(PAIRS), "handwritten.jsonl": _sha(HANDWRITTEN) if HANDWRITTEN.exists() else None,
         "generated.jsonl": _sha(GENERATED) if GENERATED.exists() else None, "seed": SEED}, indent=1))
    print(json.dumps(stats, indent=1, default=dict))


if __name__ == "__main__":
    main()
