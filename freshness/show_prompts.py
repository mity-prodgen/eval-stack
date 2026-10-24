#!/usr/bin/env python3
"""Print every post 3 request body verbatim on one real item. No API calls.

    python -m freshness.show_prompts [--item c004] [--order old_first]
"""

from __future__ import annotations

import argparse
import json

from .build_items import load_items
from .calls import build_baseline, build_decision, build_gen, build_label, systemone_url
from .common import DECIDERS, GENERATORS, JOB1_QUESTION, JOB2_QUESTION, LABELERS, NOT_REACHABLE, RAG_PROMPT_SOURCE

EXAMPLE_ANSWER = "Customers can return unused items within 30 days of delivery for a full refund."


def show(title: str, payload: dict) -> None:
    print(f"#### {title}\n{json.dumps(payload, indent=2, ensure_ascii=False)}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--item", default="c001")
    ap.add_argument("--order", default="old_first", choices=("new_first", "old_first"))
    args = ap.parse_args()
    items = {i["id"]: i for i in load_items()}
    it, o = items[args.item], args.order
    print(f"Item {it['id']} ({it['kind']}, {it['domain']}, date in {it['date_format']}), order {o}")
    print(f"question: {it['question']}\ncurrent: {it['current']}\nstale: {it['stale']}\n")
    print(f"Default RAG prompt: {RAG_PROMPT_SOURCE}")
    print(f"Job 1 question (every decision model): {JOB1_QUESTION}")
    print(f"Job 2 question (every decision model): {JOB2_QUESTION}")
    print(f"Not reachable: {NOT_REACHABLE}\n")

    for g in GENERATORS:
        show(f"generator {g}", build_gen(it, o, g).payload)
    for lb in LABELERS:
        show(f"labeler {lb}", build_label(it, EXAMPLE_ANSWER, lb).payload)
    wiki = next(i for i in items.values() if i["kind"] == "wiki")
    show(f"labeler sonnet on a WikiContradict item ({wiki['id']})", build_label(wiki, EXAMPLE_ANSWER, "sonnet").payload)
    show("faithfulness baseline (Sonnet 5.5, post 1's question)", build_baseline(it, o, EXAMPLE_ANSWER).payload)
    for d in DECIDERS:
        for job in ("job1", "job2"):
            c = build_decision(it, o, d.name, job, EXAMPLE_ANSWER)
            where = systemone_url(d.name) if d.api == "systemone" else {"jev": "TypeSafe SDK system_one",
                                                                          "luna": "OpenAI /v1/decisions"}[d.api]
            show(f"decision {d.name} / {job} -> {where}", c.payload)


if __name__ == "__main__":
    main()
