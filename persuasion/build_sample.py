#!/usr/bin/env python3
"""Build the 120-item sample: 60 answers with >=1 annotated hallucination span, 60 with none.

Stratified 3 task types (Summary/QA/Data2txt, same as post 2) x 2 classes x 20, fixed seed.
Items are drawn from post 2's 600-case sample, which is first rebuilt here with post 2's own
code and seed (jev-test@700dc5f, hallucination/build_sample.py) so the two posts share items.
If a post-2 judge_sample.jsonl is given with --post2-sample, the rebuild is checked against it
item by item, and the overlap is reported.

Writes persuasion/results/sample.jsonl (gitignored, holds passage text) and
persuasion/sample_ids.json (committed: ids and stats only).

    python persuasion/build_sample.py [--post2-sample ../jev/hallucination/results/judge_sample.jsonl]
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

from common import N_PER_CELL, POST2_SEED, RAGTRUTH, RESULTS, ROOT, SEED, TASK_TYPES

POST2_N_PER_CELL = 100
SAMPLE_FILE = RESULTS / "sample.jsonl"
IDS_FILE = ROOT / "persuasion" / "sample_ids.json"


def load_ragtruth() -> tuple[list[dict], dict[str, dict]]:
    response_path, source_path = RAGTRUTH / "response.jsonl", RAGTRUTH / "source_info.jsonl"
    if not response_path.exists() or not source_path.exists():
        raise FileNotFoundError(f"RAGTruth not found under {RAGTRUTH}. Run scripts/download_ragtruth.sh first.")
    responses = [json.loads(line) for line in response_path.open()]
    sources = {s["source_id"]: s for s in map(json.loads, source_path.open())}
    return responses, sources


def post2_sample(responses: list[dict], sources: dict[str, dict]) -> list[dict]:
    """Copy of build_cases() from jev-test@700dc5f hallucination/build_sample.py (same seed, same order)."""
    by_cell: dict[tuple[str, bool], list[dict]] = {}
    for r in responses:
        if r.get("quality") != "good":
            continue
        source = sources[r["source_id"]]
        task_type = source["task_type"]
        if task_type not in TASK_TYPES:
            continue
        hallucinated = len(r["labels"]) > 0
        case = {
            "response_id": r["id"],
            "source_id": r["source_id"],
            "task_type": task_type,
            "corpus": source["source"],
            "generating_model": r["model"],
            "source_passage": source["source_info"],
            "response": r["response"],
            "labels": r["labels"],
            "hallucinated": hallucinated,
        }
        by_cell.setdefault((task_type, hallucinated), []).append(case)

    rng = random.Random(POST2_SEED)
    sampled: list[dict] = []
    for task_type in TASK_TYPES:
        for hallucinated in (True, False):
            sampled.extend(rng.sample(by_cell[(task_type, hallucinated)], POST2_N_PER_CELL))
    rng.shuffle(sampled)
    for i, case in enumerate(sampled):
        halluc_tag = "halluc" if case["hallucinated"] else "clean"
        case["case_id"] = f"{case['task_type'].lower()}_{halluc_tag}_{i:03d}"
    return sampled


def draw(pool: list[dict], seed: int = SEED, n_per_cell: int = N_PER_CELL) -> list[dict]:
    cells: dict[tuple[str, bool], list[dict]] = collections.defaultdict(list)
    for c in pool:
        cells[(c["task_type"], c["hallucinated"])].append(c)
    rng = random.Random(seed)
    out = []
    for task_type in TASK_TYPES:
        for hallucinated in (True, False):
            cell = sorted(cells[(task_type, hallucinated)], key=lambda c: c["response_id"])
            out.extend(rng.sample(cell, n_per_cell))
    return out


def stats(sample: list[dict]) -> dict:
    halluc = [c for c in sample if c["hallucinated"]]
    return {
        "n": len(sample),
        "n_hallucinated": len(halluc),
        "n_clean": len(sample) - len(halluc),
        "by_cell": {f"{t}/{'halluc' if h else 'clean'}": sum(1 for c in sample if c["task_type"] == t and c["hallucinated"] == h)
                    for t in TASK_TYPES for h in (True, False)},
        "spans_per_hallucinated_answer": dict(sorted(collections.Counter(len(c["labels"]) for c in halluc).items())),
        "span_label_types": dict(collections.Counter(l["label_type"] for c in halluc for l in c["labels"])),
        "generating_models": dict(collections.Counter(c["generating_model"] for c in sample)),
        "response_chars_median": sorted(len(c["response"]) for c in sample)[len(sample) // 2],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--post2-sample", type=Path, help="post 2's judge_sample.jsonl, to verify the rebuild")
    args = ap.parse_args()

    responses, sources = load_ragtruth()
    p2 = post2_sample(responses, sources)

    check = None
    if args.post2_sample:
        theirs = [json.loads(line) for line in args.post2_sample.open()]
        keys = ("case_id", "response_id", "response", "labels", "hallucinated", "task_type")
        mismatches = sum(1 for a, b in zip(p2, theirs) if any(a[k] != b[k] for k in keys))
        check = {"post2_file": str(args.post2_sample), "n_theirs": len(theirs), "n_rebuilt": len(p2),
                 "field_mismatches": mismatches + abs(len(p2) - len(theirs))}
        print(f"post-2 rebuild vs {args.post2_sample}: {check['field_mismatches']} mismatching items of {len(theirs)}")
        if check["field_mismatches"]:
            raise SystemExit("post-2 sample rebuild does not match - fix before going on")

    sample = draw(p2)
    p2_ids = {c["response_id"] for c in p2}
    overlap = sum(1 for c in sample if c["response_id"] in p2_ids)

    with SAMPLE_FILE.open("w") as f:
        for c in sample:
            f.write(json.dumps(c) + "\n")
    digest = hashlib.sha256("\n".join(c["response_id"] for c in sample).encode()).hexdigest()[:16]
    out = {
        "seed": SEED,
        "post2_seed": POST2_SEED,
        "source": "RAGTruth quality=='good', drawn from post 2's 600-case sample",
        "overlap_with_post2_sample": {"n": overlap, "of": len(sample)},
        "post2_rebuild_check": check,
        "response_ids_sha256_16": digest,
        "stats": stats(sample),
        "items": [{"response_id": c["response_id"], "post2_case_id": c["case_id"], "task_type": c["task_type"],
                   "hallucinated": c["hallucinated"], "n_spans": len(c["labels"])} for c in sample],
    }
    IDS_FILE.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {len(sample)} items to {SAMPLE_FILE} (ids: {IDS_FILE})")
    print(f"overlap with post-2 600-case sample: {overlap}/{len(sample)}")
    print(json.dumps(out["stats"], indent=1))


if __name__ == "__main__":
    main()
