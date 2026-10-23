"""Paths, seeds, pinned models and every prompt for post 3 (stale documents in RAG).

Everything a reader needs to audit the wording lives in this file: the default RAG prompt and the F1/F2 additions,
the F2 pre-check questions, the labeler prompts, the J0/J1 judge rubrics, and how chunks are rendered.
`python -m conflict.show_prompts` prints every request body verbatim on a real item.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.post1_judge import JEV_P0_CRITERIA, JEV_P0_INSTRUCTIONS, LUNA_P0, judge_user_text, passage_text  # noqa: E402
from harness.post2_judge import JUDGE_SYSTEM_PROMPT as CLAUDE_J0_SYSTEM  # noqa: E402

PKG = ROOT / "conflict"
DATA_DIR = PKG / "data"                     # committed: authored doc sets, frozen items, review CSV, stats
DOCSETS = DATA_DIR / "docsets.json"
ITEMS = DATA_DIR / "items.jsonl"            # authored stale/current + control items (frozen, sha256 in MANIFEST)
WIKI_IDS = DATA_DIR / "wiki_ids.json"       # WikiContradict question_IDs + distractor ids (text not committed)
REVIEW = DATA_DIR / "review.csv"
STATS = DATA_DIR / "stats.json"
MANIFEST = DATA_DIR / "MANIFEST.json"
WIKI_RAW = ROOT / "data" / "wikicontradict" / "WikiContradict_dataset_v1_rag_qa.csv"   # scripts/download_wikicontradict.sh
RESULTS = PKG / "results"                   # gitignored: built wiki items, answers, labels, judgments, logs
FIGURES = PKG / "figures"
RESULTS_MD = PKG / "RESULTS.md"
CACHE = ROOT / "cache"                      # shared API response cache (gitignored)
CACHE_NS = "conflict"

SEED = 20261022
N_CONTROL = 40
N_WIKI = 20
N_REVIEW = 30
N_NOISE = 40           # conflict items re-run at F0 (both orders) to measure label changes between runs
N_PILOT = 10           # STOP 2: items per generator per arm (6 conflict, 2 control, 2 wiki), both orders
N_HANDCHECK = 40       # STOP 2: labels for the human hand-check

KINDS = ("conflict", "control", "wiki")
DATE_FORMATS = ("header", "body", "version")
ORDERS = ("new_first", "old_first")          # wiki: "new" = context1, "old" = context2 (neither is newer)
N_CHUNKS = 6

# =====================================================================================================
# Models (IDs and prices verified in harness/prices.py, with sources and dates)
# =====================================================================================================
GENERATORS = {"haiku55": "claude-haiku-5-5", "sonnet": "claude-sonnet-5-5", "sol": "gpt-6.1-sol",
              "qwen": "Qwen/Qwen3.8-27B"}
LABELERS = {"sonnet": "claude-sonnet-5-5", "sol": "gpt-6.1-sol"}
# "Use the labeler that is not the generator when they overlap." For the two generators that are neither labeler,
# the labeler from the other vendor (Haiku 5.5 -> Sol); Qwen gets Sonnet so each labeler is primary for two
# generators. Every result is also reported with the other labeler.
PRIMARY_LABELER = {"haiku55": "sol", "sonnet": "sol", "sol": "sonnet", "qwen": "sonnet"}
JUDGES = {"jev": "jev-1.13.0", "luna": "gpt-6-luna", "haiku45": "claude-haiku-4-5-20251001",
          "sonnet": "claude-sonnet-5-5"}
PRECHECKERS = {"jev": "jev-1.13.0", "luna": "gpt-6-luna"}
ARMS = ("F0", "F1", "F2jev", "F2luna")
RUBRICS = ("J0", "J1")
PASS_THRESHOLD = 0.5      # judges: P(pass) >= 0.5
PRECHECK_THRESHOLD = 0.5  # F2: notice if P(conflict) >= 0.5

# Generator settings: lowest reasoning each model allows, i.e. the "instruct" setting a RAG answer path usually runs.
# Claude 5.5 models reject temperature (400); Haiku 5.5 accepts thinking "disabled", Sonnet 5.5's lowest is
# "between_tools" (no thinking on tool-free requests). Sol cannot turn reasoning off; "low" is its lowest effort.
# Qwen3.8 thinks by default; enable_thinking=False is the documented instruct mode, with the model card's
# recommended instruct sampling.
GEN_MAX_TOKENS = 1024
CLAUDE_GEN_SETTINGS = {"claude-haiku-5-5": {"thinking": {"type": "disabled"}},
                       "claude-sonnet-5-5": {"thinking": {"type": "between_tools"}}}
SOL_GEN_EFFORT = "low"
DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"
QWEN_SAMPLING = {"temperature": 0.7, "top_p": 0.8, "presence_penalty": 1.5}
QWEN_EXTRA_BODY = {"top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0,
                   "chat_template_kwargs": {"enable_thinking": False}}

LABEL_MAX_TOKENS = 100
SOL_LABEL_EFFORT = "low"
CLAUDE_LABEL_SETTINGS = {"thinking": {"type": "between_tools"}}
JUDGE_MAX_TOKENS = 200   # post 2 / post 1 value
CLAUDE_JUDGE_SETTINGS = {"claude-haiku-4-5-20251001": {"temperature": 0},          # post 1 anchor settings
                         "claude-sonnet-5-5": {"thinking": {"type": "between_tools"}, "cache_passage": True}}

# =====================================================================================================
# Chunk rendering. One doc set = one organisation's document in two versions; each section is one chunk.
# =====================================================================================================
HEADER_STYLES = {"effective": "Effective {long}", "updated": "Last updated: {iso}", "published": "Published {dmy}"}


def _d(iso: str) -> date:
    return date.fromisoformat(iso)


def long_date(iso: str) -> str:
    d = _d(iso)
    return f"{d.strftime('%B')} {d.day}, {d.year}"


def dmy_date(iso: str) -> str:
    d = _d(iso)
    return f"{d.day} {d.strftime('%B')} {d.year}"


def month_year(iso: str) -> str:
    return _d(iso).strftime("%B %Y")


def _lower_first(text: str, org: str) -> str:
    """'Unused items ...' -> 'unused items ...' after 'As of March 2026, '; proper nouns and acronyms stay."""
    first = text.split()[0]
    if first in org.split() or first[:2].isupper() or first.startswith("/"):
        return text
    return text[0].lower() + text[1:]


def fill(text: str, a: str, b: str) -> str:
    return text.replace("{a}", a).replace("{b}", b)


def render_chunk(ds: dict, sec: dict, version: str, date_format: str, a: str, b: str) -> str:
    """One chunk. The date appears in a header line, in the body ("As of March 2026, ..."), or only as a version."""
    meta = ds[version]
    body = fill(sec["text"], a, b)
    title = f"{ds['org']} | {ds['title']}"
    if date_format == "header":
        line = HEADER_STYLES[ds["header_style"]].format(long=long_date(meta["date"]), iso=meta["date"],
                                                        dmy=dmy_date(meta["date"]))
        return f"{title}\n{line}\n{sec['heading']}\n{body}"
    if date_format == "body":
        return f"{title}\n{sec['heading']}\nAs of {month_year(meta['date'])}, {_lower_first(body, ds['org'])}"
    if date_format == "version":
        return f"{title} ({meta['version']})\n{sec['heading']}\n{body}"
    raise ValueError(date_format)


def wiki_chunk(title: str, text: str) -> str:
    return f"{title}\n{text.strip()}"


def context_text(chunks: list[str]) -> str:
    """Numbered sources, so the F2 notice can name one. (LangChain's format_docs joins with a blank line and no
    numbers; the numbering is the one change.)"""
    return "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(chunks))


# =====================================================================================================
# Generation: default RAG prompt (F0) and the two generation-side fixes (F1, F2)
# =====================================================================================================
# LangChain Hub rlm/rag-prompt, verbatim: commit 50442af133e61576e74536c6556cefe1fac147cad032f4377b60c436e6cdcb6e,
# fetched 2026-10-22 from https://api.smith.langchain.com/commits/rlm/rag-prompt/latest. One human message.
RAG_PROMPT_SOURCE = "LangChain Hub rlm/rag-prompt @ 50442af1 (fetched 2026-10-22)"
RAG_PROMPT = (
    "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep "
    "the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"
)
_INSERT_AFTER = "keep the answer concise."
assert _INSERT_AFTER in RAG_PROMPT

F1_RULE = "If sources disagree, use the newest dated source and say that an older version exists."
F2_NOTICE = "Note: the retrieved sources give different answers to this question."
F2_NOTICE_NEWEST = " Source [{k}] is the newest version."


def f2_notice(newest: str | None) -> str:
    return F2_NOTICE + (F2_NOTICE_NEWEST.format(k=newest) if newest and newest.isdigit() else "")


def gen_prompt(question: str, context: str, arm: str, notice: str | None = None) -> str:
    """F0: the template. F1: + the rule. F2: + a notice, only when the pre-check fires (else the F0 prompt)."""
    extra = ""
    if arm == "F1":
        extra = " " + F1_RULE
    elif arm.startswith("F2") and notice:
        extra = " " + notice
    elif arm not in ARMS:
        raise ValueError(arm)
    tmpl = RAG_PROMPT.replace(_INSERT_AFTER, _INSERT_AFTER + extra, 1)
    return tmpl.format(question=question, context=context)


# =====================================================================================================
# F2 pre-check: "Do these chunks give different answers to the question?" + which source is newest
# =====================================================================================================
PRECHECK_QUESTION = "Do these chunks give different answers to the question?"
_NEWEST = ("Among the sources that answer the question, which one is the newest version, judging by the dates or "
           "version numbers shown in the sources?")
_NONE = "No source that answers the question shows a date or version number that can be compared."

JEV_PRECHECK = {
    "conflict": {
        "type": "noul",
        "instructions": "The chunks in `sources` give different answers to `question`.",
        "criteria": {
            "true": "At least two chunks answer `question` and their answers differ, for example a different number, "
                    "date, limit or condition.",
            "false": "Every chunk that answers `question` gives the same answer, or at most one chunk answers it.",
        },
    },
    "newest": {
        "type": "choice",
        "instructions": _NEWEST.replace("the sources that answer the question", "the chunks in `sources` that answer "
                                        "`question`").replace("shown in the sources", "shown in the chunks"),
        "criteria": {**{str(i): f"Source [{i}]" for i in range(1, N_CHUNKS + 1)}, "none": _NONE},
    },
}
LUNA_PRECHECK = [
    {"type": "predicate", "name": "conflict",
     "instructions": PRECHECK_QUESTION + " True if at least two SOURCES answer the QUESTION and their answers differ "
                     "(for example a different number, date, limit or condition). False if every source that answers "
                     "it gives the same answer, or at most one source answers it."},
    {"type": "choice", "name": "newest", "instructions": _NEWEST,
     "choices": [*({"value": str(i), "description": f"Source [{i}]"} for i in range(1, N_CHUNKS + 1)),
                 {"value": "none", "description": _NONE}]},
]


def precheck_state(question: str, context: str) -> dict[str, str]:
    return {"question": question, "sources": context}


def precheck_input(question: str, context: str) -> str:
    return f"QUESTION:\n{question}\n\nSOURCES:\n{context}"


# =====================================================================================================
# Answer labels (two labelers: Claude Sonnet 5.5 and gpt-6.1-sol, same prompt)
# =====================================================================================================
LABELS = ("current", "stale", "blends", "flags", "refuses", "other")
WIKI_LABELS = ("a", "b", "blends", "flags", "refuses", "other")

_LABEL_FORMAT = (
    "Judge only what the answer says. Small differences in wording, units or number format do not matter (for "
    'example "14 days" and "two weeks" are the same value).\n\n'
    "Your entire reply must be exactly one line of JSON and nothing else: "
    '{"label": "<one of the labels>", "conflict_noted": true|false}'
)
LABEL_SYSTEM = (
    "You label answers written by a question-answering assistant. The assistant answered a QUESTION from retrieved "
    "documents that contained two versions of the same document: the newest version gives the CURRENT VALUE and an "
    "older version gives the OUTDATED VALUE.\n\n"
    "Choose exactly one label for the ANSWER:\n"
    "- current: the answer gives the CURRENT VALUE as its answer. It may also mention the OUTDATED VALUE as old or "
    "replaced.\n"
    "- stale: the answer gives the OUTDATED VALUE as its answer. It may also mention the CURRENT VALUE as old or "
    "replaced.\n"
    "- blends: the answer states both values as if both were true, merges them into one value, or gives both without "
    "choosing one and without saying that the sources disagree.\n"
    "- flags: the answer says that the sources disagree or conflict and does not commit to one value as its answer.\n"
    "- refuses: the answer gives no value: it says it does not know, that the context does not say, or it declines.\n"
    "- other: the answer gives a value that is neither the CURRENT VALUE nor the OUTDATED VALUE.\n\n"
    "Also set conflict_noted to true if the answer says or implies that the sources disagree, that the value changed, "
    "or that an older or newer version exists; otherwise false. When both versions give the same value, the label is "
    "current if the answer gives that value.\n\n" + _LABEL_FORMAT
)
WIKI_LABEL_SYSTEM = (
    "You label answers written by a question-answering assistant. The assistant answered a QUESTION from retrieved "
    "passages. Two of the passages contradict each other: one supports ANSWER A and the other supports ANSWER B.\n\n"
    "Choose exactly one label for the ANSWER:\n"
    "- a: the answer gives ANSWER A as its answer and does not present ANSWER B as also possible.\n"
    "- b: the answer gives ANSWER B as its answer and does not present ANSWER A as also possible.\n"
    "- blends: the answer states both as if both were true or merges them, without saying the passages disagree.\n"
    "- flags: the answer says that the passages disagree, conflict or give different answers.\n"
    "- refuses: the answer gives no answer: it says it does not know, that the context does not say, or it declines.\n"
    "- other: the answer gives something that is neither ANSWER A nor ANSWER B.\n\n"
    "Also set conflict_noted to true if the answer says or implies that the passages disagree; otherwise false.\n\n"
    + _LABEL_FORMAT
)


def label_user_text(item: dict, answer: str) -> str:
    if item["kind"] == "wiki":
        return (f"QUESTION:\n{item['question']}\n\nANSWER A:\n{item['answer_a']}\n\nANSWER B:\n{item['answer_b']}"
                f"\n\nANSWER:\n{answer}")
    outdated = item["stale"] + (" (both versions give the same value)" if item["kind"] == "control" else "")
    return (f"QUESTION:\n{item['question']}\n\nCURRENT VALUE:\n{item['current']}\n\nOUTDATED VALUE:\n{outdated}"
            f"\n\nANSWER:\n{answer}")


def label_system(item: dict) -> str:
    return WIKI_LABEL_SYSTEM if item["kind"] == "wiki" else LABEL_SYSTEM


# =====================================================================================================
# Judges: post 1's P0 faithfulness question (J0) and the J1 rubric fix. Run on F0 answers.
# =====================================================================================================
J1_SENTENCE = "If the sources disagree, the answer must use the newest source or say that the sources disagree."

JEV_J0 = {"supported": {"type": "noul", "instructions": JEV_P0_INSTRUCTIONS, "criteria": JEV_P0_CRITERIA}}
JEV_J1 = {"supported": {
    "type": "noul",
    "instructions": JEV_P0_INSTRUCTIONS + " " + J1_SENTENCE,
    "criteria": {
        "true": JEV_P0_CRITERIA["true"] + " If the sources in `source_passage` disagree, `response` uses the newest "
                "source or says that the sources disagree.",
        "false": JEV_P0_CRITERIA["false"] + " Or the sources in `source_passage` disagree and `response` neither uses "
                 "the newest source nor says that the sources disagree.",
    },
}}
LUNA_J0 = [{"type": "predicate", "name": "supported", "instructions": LUNA_P0}]
LUNA_J1 = [{"type": "predicate", "name": "supported", "instructions": LUNA_P0 + " " + J1_SENTENCE}]

_FORMAT_PARAGRAPH_START = "Your entire reply must be"
assert _FORMAT_PARAGRAPH_START in CLAUDE_J0_SYSTEM
CLAUDE_J1_SYSTEM = CLAUDE_J0_SYSTEM.replace(
    _FORMAT_PARAGRAPH_START,
    J1_SENTENCE + " Answer hallucinated=true if the sources disagree and the response does neither.\n\n"
    + _FORMAT_PARAGRAPH_START, 1)


def judge_passage(item: dict, context: str) -> dict[str, str]:
    """Same shape as post 1's RAGTruth QA passages: {"question", "passages"}, rendered as indented JSON."""
    return {"question": item["question"], "passages": context}


__all__ = ["judge_user_text", "passage_text"]
