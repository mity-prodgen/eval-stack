#!/usr/bin/env python3
"""Build the frozen item set from the authored doc sets and the WikiContradict sample. Deterministic (seeded).

  conflict   one per section of data/docsets.json (101): the question's value differs between the old and the new
             version of the section. The other value in the section ({b}) also changes, as in a real revision.
  control    40 of the same questions (stratified by domain): the old version gives the CURRENT value for the asked
             slot and keeps its old value for the other slot, so the two chunks still differ, but not on the answer.
  wiki       20 WikiContradict instances (MIT, IBM, NeurIPS 2024), stratified by contradiction type. Only ids are
             committed (data/wiki_ids.json); the text is rebuilt from the pinned download.

Every item has 6 chunks: the 2 key chunks (old and new version of the section, or WikiContradict's two contexts)
and 4 distractors from the same doc set (other sections, each in a random version; wiki: contexts from other
articles). Chunk order is set at run time: key chunks at ranks 1 and 2 in both orders, distractors at 3-6.

Doc sets were written by Claude Opus 5.5 in the authoring session (no API call); Opus 5.5 is not a generator,
labeler or judge in this test.

    python -m freshness.build_items             # check that the frozen files rebuild byte for byte
    python -m freshness.build_items --refreeze  # rewrite them
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import statistics
from collections import Counter, defaultdict

from .common import (DATA_DIR, DATE_FORMATS, DOCSETS, ITEMS, MANIFEST, N_CONTROL, N_REVIEW, N_WIKI, REVIEW, SEED,
                     STATS, WIKI_IDS, WIKI_RAW, render_chunk, wiki_chunk)

N_DISTRACTORS = 4


def has(value: str, text: str) -> bool:
    """value occurs in text as a whole token ('8' is not in '28 days', '16' is not in '16.2')."""
    return re.search(r"(?<![\w.$€])" + re.escape(value) + r"(?![\w]|\.\d)", text) is not None


def load_docsets() -> list[dict]:
    return json.loads(DOCSETS.read_text())


# ------------------------------------------------------------------------------------------- checks
def check_docsets(docsets: list[dict]) -> list[str]:
    """Problems that would make an item ambiguous. Empty list = fine."""
    problems = []
    for ds in docsets:
        if len(ds["sections"]) < N_DISTRACTORS + 1:
            problems.append(f"{ds['id']}: needs >= {N_DISTRACTORS + 1} sections")
        if not ds["old"]["date"] < ds["new"]["date"]:
            problems.append(f"{ds['id']}: old date not before new date")
        for sec in ds["sections"]:
            sid = f"{ds['id']}/{sec['id']}"
            (a_old, a_new), (b_old, b_new) = sec["a"], sec["b"]
            if "{a}" not in sec["text"] or "{b}" not in sec["text"]:
                problems.append(f"{sid}: text needs {{a}} and {{b}}")
            if a_old == a_new or b_old == b_new:
                problems.append(f"{sid}: a value or b value does not change")
            if a_old in a_new or a_new in a_old:
                problems.append(f"{sid}: one a value contains the other")
            for v in (a_old, a_new):
                other_text = sec["text"].replace("{a}", " ").replace("{b}", " ")
                if has(v, other_text) or has(v, b_old) or has(v, b_new) or has(b_old, v) or has(b_new, v):
                    problems.append(f"{sid}: a value {v!r} also appears elsewhere in the section")
    return problems


def check_item(it: dict) -> list[str]:
    """The value the question asks for must appear only where it should."""
    p = []
    old, new, ds = it["chunks"]["old"], it["chunks"]["new"], it["chunks"]["distractors"]
    if not has(it["current"], new):
        p.append("current value missing from new chunk")
    if it["kind"] == "conflict":
        if not has(it["stale"], old):
            p.append("stale value missing from old chunk")
        if has(it["current"], old) or has(it["stale"], new):
            p.append("value leak between versions")
    else:
        if not has(it["current"], old):
            p.append("control: old chunk lacks the shared value")
        if old == new:
            p.append("control: chunks identical")
    for d in ds:
        for v in (it["current"], it["stale"]):
            if has(v, d):
                p.append(f"value {v!r} in distractor: {d.splitlines()[-1][:90]}")
    return p


# ------------------------------------------------------------------------------------------- build
def build_authored(docsets: list[dict]) -> list[dict]:
    rng = random.Random(SEED)
    by_domain = defaultdict(list)
    for ds in docsets:
        for sec in ds["sections"]:
            by_domain[ds["domain"]].append((ds, sec))

    # date format: balanced within each domain (shuffled round-robin), so format is not confounded with domain
    fmt = {}
    offset = 0
    for dom in sorted(by_domain):
        secs = by_domain[dom][:]
        rng.shuffle(secs)
        for i, (ds, sec) in enumerate(secs):
            fmt[(ds["id"], sec["id"])] = DATE_FORMATS[(i + offset) % 3]
        offset += len(secs)

    # controls: N_CONTROL questions, spread over domains as evenly as possible
    doms = sorted(by_domain)
    quota = {d: N_CONTROL // len(doms) + (1 if i < N_CONTROL % len(doms) else 0) for i, d in enumerate(doms)}
    control_keys = set()
    for d in doms:
        pick = rng.sample(sorted(by_domain[d], key=lambda x: (x[0]["id"], x[1]["id"])), quota[d])
        control_keys |= {(ds["id"], sec["id"]) for ds, sec in pick}

    items = []
    n_c = n_k = 0
    for ds in docsets:
        for sec in ds["sections"]:
            key = (ds["id"], sec["id"])
            f = fmt[key]
            others = [s for s in ds["sections"] if s["id"] != sec["id"]]
            picks = rng.sample(others, N_DISTRACTORS)
            versions = [rng.choice(("old", "new")) for _ in picks]
            distractors = [render_chunk(ds, s, v, f, s["a"][v == "new"], s["b"][v == "new"])
                           for s, v in zip(picks, versions)]
            dmeta = [{"section": s["id"], "version": v} for s, v in zip(picks, versions)]
            (a_old, a_new), (b_old, b_new) = sec["a"], sec["b"]
            base = {"domain": ds["domain"], "docset": ds["id"], "section": sec["id"], "date_format": f,
                    "question": sec["question"], "current": a_new, "old_date": ds["old"]["date"],
                    "new_date": ds["new"]["date"], "old_version": ds["old"]["version"],
                    "new_version": ds["new"]["version"], "distractor_meta": dmeta}
            n_c += 1
            cid = f"c{n_c:03d}"
            items.append({"id": cid, "kind": "conflict", **base, "stale": a_old,
                          "chunks": {"new": render_chunk(ds, sec, "new", f, a_new, b_new),
                                     "old": render_chunk(ds, sec, "old", f, a_old, b_old),
                                     "distractors": distractors}})
            if key in control_keys:
                n_k += 1
                items.append({"id": f"k{n_k:03d}", "kind": "control", "twin": cid, **base, "stale": a_new,
                              "chunks": {"new": render_chunk(ds, sec, "new", f, a_new, b_new),
                                         "old": render_chunk(ds, sec, "old", f, a_new, b_old),
                                         "distractors": distractors}})
    return items


def load_wiki_rows() -> list[dict]:
    with WIKI_RAW.open(newline="") as f:
        return list(csv.DictReader(f))


def sample_wiki(rows: list[dict]) -> list[dict]:
    """N_WIKI instances, stratified by contradiction type (proportional), with two distinct non-empty answers and
    two distinct contexts (some instances put both sides of the contradiction in one passage; those would make the
    two chunk orders identical)."""
    rng = random.Random(SEED + 1)
    ok = [r for r in rows if r["context1"].strip() and r["context2"].strip() and r["answer1"].strip()
          and r["answer2"].strip() and r["answer1"].strip().lower() != r["answer2"].strip().lower()
          and r["context1"].strip() != r["context2"].strip()]
    by_type = defaultdict(list)
    for r in ok:
        by_type[r["contradictType"]].append(r)
    n_exp = round(N_WIKI * len(by_type["Explicit"]) / len(ok))
    quota = {"Explicit": n_exp, "Implicit (reasoning required)": N_WIKI - n_exp}
    out = []
    for t in sorted(quota):
        out += rng.sample(sorted(by_type[t], key=lambda r: int(r["question_ID"])), quota[t])
    picked_titles = {r["WikipediaArticleTitle"] for r in out}
    pool = sorted((r for r in ok if r["WikipediaArticleTitle"] not in picked_titles), key=lambda r: int(r["question_ID"]))
    spec = []
    for i, r in enumerate(sorted(out, key=lambda r: int(r["question_ID"]))):
        ds = rng.sample(pool, N_DISTRACTORS)
        spec.append({"id": f"w{i + 1:03d}", "question_ID": r["question_ID"], "contradictType": r["contradictType"],
                     "distractors": [{"question_ID": d["question_ID"], "context": rng.choice((1, 2))} for d in ds]})
    return spec


def wiki_items(spec: list[dict], rows: list[dict]) -> list[dict]:
    by_id = {r["question_ID"]: r for r in rows}
    items = []
    for s in spec:
        r = by_id[s["question_ID"]]
        t = r["WikipediaArticleTitle"]
        items.append({"id": s["id"], "kind": "wiki", "domain": "wikipedia", "date_format": "none",
                      "question_ID": r["question_ID"], "contradict_type": r["contradictType"],
                      "question": r["question"].strip(), "answer_a": r["answer1"].strip(), "answer_b": r["answer2"].strip(),
                      "current": r["answer1"].strip(), "stale": r["answer2"].strip(),
                      "chunks": {"new": wiki_chunk(t, r["context1"]), "old": wiki_chunk(t, r["context2"]),
                                 "distractors": [wiki_chunk(by_id[d["question_ID"]]["WikipediaArticleTitle"],
                                                            by_id[d["question_ID"]][f"context{d['context']}"])
                                                 for d in s["distractors"]]}})
    return items


def load_items(include_wiki: bool = True) -> list[dict]:
    """Frozen authored items + WikiContradict items rebuilt from the pinned download."""
    items = [json.loads(l) for l in ITEMS.open()]
    if include_wiki:
        if not WIKI_RAW.exists():
            raise SystemExit(f"{WIKI_RAW} missing - run scripts/download_wikicontradict.sh")
        items += wiki_items(json.loads(WIKI_IDS.read_text())["items"], load_wiki_rows())
    return items


# ------------------------------------------------------------------------------------------- outputs
def review_rows(items: list[dict]) -> list[dict]:
    rng = random.Random(SEED + 2)
    pool = [it for it in items if it["kind"] in ("conflict", "control")]
    out = []
    for it in sorted(rng.sample(pool, N_REVIEW), key=lambda x: x["id"]):
        out.append({"id": it["id"], "kind": it["kind"], "domain": it["domain"], "date_format": it["date_format"],
                    "question": it["question"], "current_answer": it["current"],
                    "stale_answer": it["stale"] if it["kind"] == "conflict" else "(same as current)",
                    "new_chunk": it["chunks"]["new"], "old_chunk": it["chunks"]["old"],
                    "reviewer_ok (y/n)": "", "reviewer_note": ""})
    return out


def stats(items: list[dict], wiki_spec: list[dict]) -> dict:
    words = [len(c.split()) for it in items for c in (it["chunks"]["new"], it["chunks"]["old"], *it["chunks"]["distractors"])]
    conflict = [it for it in items if it["kind"] == "conflict"]
    gap = [(__import__("datetime").date.fromisoformat(it["new_date"]) -
            __import__("datetime").date.fromisoformat(it["old_date"])).days for it in conflict]
    return {
        "n_items": len(items) + len(wiki_spec),
        "by_kind": {**Counter(it["kind"] for it in items), "wiki": len(wiki_spec)},
        "conflict_by_domain": dict(sorted(Counter(it["domain"] for it in conflict).items())),
        "conflict_by_date_format": dict(sorted(Counter(it["date_format"] for it in conflict).items())),
        "control_by_domain": dict(sorted(Counter(it["domain"] for it in items if it["kind"] == "control").items())),
        "control_by_date_format": dict(sorted(Counter(it["date_format"] for it in items if it["kind"] == "control").items())),
        "wiki_by_type": dict(Counter(s["contradictType"] for s in wiki_spec)),
        "doc_sets": len({it["docset"] for it in conflict}),
        "words_per_chunk_authored": {"median": statistics.median(words), "min": min(words), "max": max(words)},
        "days_between_versions": {"median": statistics.median(gap), "min": min(gap), "max": max(gap)},
        "runs_per_item": "2 chunk orders (new_first, old_first)",
        "prompts_per_generator_per_arm": 2 * (len(items) + len(wiki_spec)),
    }


def sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refreeze", action="store_true")
    args = ap.parse_args()

    docsets = load_docsets()
    problems = check_docsets(docsets)
    items = build_authored(docsets)
    for it in items:
        problems += [f"{it['id']}: {p}" for p in check_item(it)]
    if problems:
        raise SystemExit("data problems:\n  " + "\n  ".join(problems))
    rows = load_wiki_rows()
    wiki_spec = sample_wiki(rows)
    witems = wiki_items(wiki_spec, rows)

    items_text = "".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items)
    wiki_text = json.dumps({"source": "ibm-research/Wikipedia_contradict_benchmark @ c20e361f985ed480a659b35d98b49f2311fcd174",
                            "file": "WikiContradict_dataset_v1_rag_qa.csv", "seed": SEED + 1, "items": wiki_spec},
                           indent=1) + "\n"
    if not args.refreeze:
        same = ITEMS.exists() and ITEMS.read_text() == items_text and WIKI_IDS.read_text() == wiki_text
        print(f"{len(items)} authored items + {len(witems)} wiki items; frozen files "
              + ("rebuild byte for byte" if same else "DIFFER - pass --refreeze to rewrite them"))
        if not same:
            raise SystemExit(1)
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ITEMS.write_text(items_text)
    WIKI_IDS.write_text(wiki_text)
    rr = review_rows(items)
    with REVIEW.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rr[0]))
        w.writeheader()
        w.writerows(rr)
    st = stats(items, wiki_spec)
    STATS.write_text(json.dumps(st, indent=1) + "\n")
    MANIFEST.write_text(json.dumps({"docsets.json": sha(DOCSETS), "items.jsonl": sha(ITEMS),
                                    "wiki_ids.json": sha(WIKI_IDS), "seed": SEED}, indent=1) + "\n")
    print(json.dumps(st, indent=1))
    print(f"wrote {ITEMS}, {WIKI_IDS}, {REVIEW}, {STATS}, {MANIFEST}")


if __name__ == "__main__":
    main()
