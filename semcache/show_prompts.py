#!/usr/bin/env python3
"""Print every A3 request body verbatim (one example pair), plus each embedding model's request settings.

    python -m semcache.show_prompts
"""

from __future__ import annotations

import json

from .common import EMBEDDERS, GEMINI_SIMILARITY_PREFIX
from .decide import plan

EXAMPLE = {"id": "example", "cached": "Cheapest flights from DXB to LHR next month",
           "new": "Cheapest flights from LHR to DXB next month"}


def main() -> None:
    print("=== Embedding requests ===")
    for e in EMBEDDERS:
        print(f"- {e.name}: {e.provider} {e.model_id}{' @ ' + e.revision if e.revision else ''} | {e.settings}")
    print(f"  (gemini text sent = {GEMINI_SIMILARITY_PREFIX!r} + query, each text in its own Content)")
    for d in plan([EXAMPLE], ["jev", "luna"]):
        print(f"\n=== A3 {d.decider} ({d.model}) request body ===")
        print(json.dumps(d.payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
