"""Post 1's baseline (P0) judge wording for Jev and GPT-6 Luna, and the helpers every post's judges share.

Moved here from persuasion/common.py (which re-exports them unchanged) so later posts reuse the exact post 1
faithfulness question - "Is every claim in the answer supported by the context?" - without importing post 1's
module. The Claude P0 prompt is post 2's and lives in harness/post2_judge.py.
"""

from __future__ import annotations

import json
from typing import Any

# Jev (TypeSafe Noul question). Post 2's question, verbatim.
JEV_P0_INSTRUCTIONS = (
    "Every factual claim in `response` is directly supported by `source_passage`. "
    "`true` requires every claim in `response` to trace to an explicit statement in "
    "`source_passage`, with nothing added or contradicted. `false` if any claim in "
    "`response` contradicts `source_passage`, states a number, date, or name not "
    "present in `source_passage`, or adds a specific detail `source_passage` does not support."
)
JEV_P0_CRITERIA = {
    "true": "Every claim in `response` can be traced to an explicit statement in `source_passage`.",
    "false": (
        "At least one claim in `response` contradicts `source_passage`, or states/adds "
        "a specific detail not present in it."
    ),
}

# GPT-6 Luna (OpenAI Decisions API predicate question), mirroring post 2's wording.
LUNA_P0 = (
    "Is every factual claim in the RESPONSE directly supported by the SOURCE PASSAGE? True only if every "
    "claim in the RESPONSE traces to an explicit statement in the SOURCE PASSAGE, with nothing added or "
    "contradicted. False if any claim contradicts the SOURCE PASSAGE, states a number, date, or name not "
    "present in it, or adds a specific detail it does not support."
)


def passage_text(passage: Any) -> str:
    """Same rendering post 2 used for the LLM judge (dict passages -> indented JSON)."""
    return passage if isinstance(passage, str) else json.dumps(passage, indent=2)


def judge_user_text(passage: Any, response: str) -> str:
    """Post 2's LLM-judge user message, byte for byte. Luna gets the same text as `input`."""
    return f"SOURCE PASSAGE:\n{passage_text(passage)}\n\nRESPONSE:\n{response}"


def p_from_verdict(verdict_true: bool, confidence01: float) -> float:
    """Map (verdict, 0-1 confidence in that verdict, 0 = coin flip) to P(verdict-question is true).

    Post 2's prompt defines confidence as certainty in the stated verdict with 0 = coin flip,
    so P = 0.5 + 0.5*c for a true verdict and 0.5 - 0.5*c for a false one.
    """
    c = min(max(confidence01, 0.0), 1.0)
    return 0.5 + 0.5 * c if verdict_true else 0.5 - 0.5 * c
