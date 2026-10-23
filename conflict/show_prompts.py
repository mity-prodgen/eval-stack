#!/usr/bin/env python3
"""Print every post 3 request body verbatim on one real item. No API calls.

    python -m conflict.show_prompts [--item c004] [--order old_first]
"""

from __future__ import annotations

import argparse
import json

from .build_items import load_items
from .calls import build_gen, build_judge, build_label, build_precheck
from .common import (F1_RULE, GENERATORS, JUDGES, LABELERS, PRECHECKERS, PRIMARY_LABELER, RAG_PROMPT_SOURCE, RUBRICS,
                     f2_notice)

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
    print(f"Default RAG prompt: {RAG_PROMPT_SOURCE}\nF1 rule: {F1_RULE}\nF2 notice: {f2_notice('2')!r} "
          f"(without the second sentence when the pre-checker picks 'none')\n")

    for w in PRECHECKERS:
        show(f"F2 pre-check / {w}", build_precheck(it, o, w).payload)
    for g in GENERATORS:
        show(f"generator {g} / F0", build_gen(it, o, g, "F0").payload)
    show("generator sonnet / F1 (only the instruction line differs for the other generators)",
         build_gen(it, o, "sonnet", "F1").payload)
    show("generator sonnet / F2 with a notice", build_gen(it, o, "sonnet", "F2jev", f2_notice("2")).payload)
    for lb in LABELERS:
        show(f"labeler {lb} (primary for: {[g for g, p in PRIMARY_LABELER.items() if p == lb]})",
             build_label(it, EXAMPLE_ANSWER, lb).payload)
    wiki = next(i for i in items.values() if i["kind"] == "wiki")
    show(f"labeler sonnet on a WikiContradict item ({wiki['id']})", build_label(wiki, EXAMPLE_ANSWER, "sonnet").payload)
    for j in JUDGES:
        for r in RUBRICS:
            show(f"judge {j} / {r}", build_judge(it, o, EXAMPLE_ANSWER, j, r).payload)


if __name__ == "__main__":
    main()
