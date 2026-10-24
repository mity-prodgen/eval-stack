"""A spend ledger that turns a per-command --max-usd gate into a hard cap on a whole post.

Every live command appends what it actually spent (from the APIs' own usage numbers, uncached calls only) to a
JSON file in the post's gitignored results/ folder. Before any call, a command's estimate must fit in
min(--max-usd, cap - spent so far); otherwise nothing runs. Cached calls cost nothing and are never counted.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


class Ledger:
    def __init__(self, path: Path, cap_usd: float) -> None:
        self.path, self.cap = Path(path), cap_usd

    def entries(self) -> list[dict]:
        return json.loads(self.path.read_text())["entries"] if self.path.exists() else []

    def spent(self) -> float:
        return sum(e["usd"] for e in self.entries())

    def remaining(self) -> float:
        return self.cap - self.spent()

    def allowance(self, max_usd: float) -> float:
        return min(max_usd, self.remaining())

    def record(self, command: str, step: str, usd: float, estimated_usd: float | None = None) -> None:
        entries = self.entries()
        entries.append({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "command": command, "step": step,
                        "usd": round(usd, 6), "estimated_usd": None if estimated_usd is None else round(estimated_usd, 6)})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"cap_usd": self.cap, "entries": entries}, indent=1) + "\n")
