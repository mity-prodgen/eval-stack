#!/usr/bin/env python3
"""Print every prompt and request shape verbatim, rendered on one example item. No API calls.

    python persuasion/show_prompts.py [--item RESPONSE_ID]
"""

from __future__ import annotations

import argparse
import json

from common import (APPENDED, APPEND_SEPARATOR, ARMS, JUDGE_MODELS, REWRITE_INSTRUCTIONS, RESULTS, rewrite_user_text)
from judges import BUILDERS, Call


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--item", default=None)
    args = ap.parse_args()
    sample = [json.loads(l) for l in (RESULTS / "sample.jsonl").open()]
    it = next((s for s in sample if s["response_id"] == args.item), None) or next(s for s in sample if s["hallucinated"])
    short = it["response"][:160] + " [...]"
    passage = "<SOURCE PASSAGE>"

    print("#### Conditions (appended after", repr(APPEND_SEPARATOR), ")")
    for k, v in APPENDED.items():
        print(f"{k}: {v!r}")
    print("C4: frozen rewrite from persuasion/rewrites/rewrites_v1.jsonl\n")

    for j in ("jev", "luna", "haiku", "sonnet"):
        for arm in ARMS:
            kinds = [("guard", "guard")] if (j in ("haiku", "sonnet") and arm == "P0G") else [(arm, "judge")]
            for a, kind in kinds:
                c = BUILDERS[j](Call(j, JUDGE_MODELS[j], a, kind, it["response_id"], "C0", passage, short))
                label = f"{j} / {arm}" + (" (separate guard request; P0G = P0 judgment + this)" if kind == "guard" else "")
                print(f"#### {label}\n{json.dumps(c.payload, indent=2)}\n")

    print("#### C4 rewrite request (gpt-6.1-sol, reasoning effort low)")
    print("instructions:\n" + REWRITE_INSTRUCTIONS)
    print("\ninput:\n" + rewrite_user_text(short, [l["text"] for l in it["labels"]]))


if __name__ == "__main__":
    main()
