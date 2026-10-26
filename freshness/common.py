"""Paths, seeds, pinned models and every prompt and question for post 3 (decision models vs stale RAG answers).

Everything a reader needs to audit the wording lives here: the default RAG prompt, the labeler prompts, the
faithfulness baseline, the two decision-model questions (identical for every model), how chunks and states are
rendered, and the decision-model registry. `python -m freshness.show_prompts` prints every request body verbatim.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.post2_judge import JUDGE_SYSTEM_PROMPT as BASELINE_SYSTEM  # noqa: E402

PKG = ROOT / "freshness"
DATA_DIR = PKG / "data"                     # committed: authored doc sets, frozen items, review CSV, stats
DOCSETS = DATA_DIR / "docsets.json"
ITEMS = DATA_DIR / "items.jsonl"            # authored stale/current + control items (frozen, sha256 in MANIFEST)
WIKI_IDS = DATA_DIR / "wiki_ids.json"       # WikiContradict question_IDs + distractor ids (text not committed)
REVIEW = DATA_DIR / "review.csv"
STATS = DATA_DIR / "stats.json"
MANIFEST = DATA_DIR / "MANIFEST.json"
HANDCHECK = DATA_DIR / "handcheck.csv"      # STOP 2: 40 labels for the human hand-check
WIKI_RAW = ROOT / "data" / "wikicontradict" / "WikiContradict_dataset_v1_rag_qa.csv"   # scripts/download_wikicontradict.sh
RESULTS = PKG / "results"                   # gitignored: answers, labels, decisions, latency, spend ledger
FIGURES = PKG / "figures"
RESULTS_MD = PKG / "RESULTS.md"
CACHE = ROOT / "cache"                      # shared API response cache (gitignored)
CACHE_NS = "freshness"

SEED = 20261022        # the item set was frozen with this seed; changing it changes the data
N_CONTROL = 40
N_WIKI = 20
N_REVIEW = 30
N_NOISE = 40           # conflict items asked twice (both jobs, both orders) to measure verdict changes
N_PILOT = 10           # STOP 2: 6 conflict, 2 control, 2 wiki, both orders
N_HANDCHECK = 40
N_LATENCY = 100        # sequential Job 1 calls per decision model for the latency table (after 5 warm-up calls)
N_LATENCY_WARMUP = 5
TOTAL_USD_CAP = 5.0    # hard cap on the whole post, enforced across commands by the spend ledger

KINDS = ("conflict", "control", "wiki")
DATE_FORMATS = ("header", "body", "version")
ORDERS = ("new_first", "old_first")          # wiki: "new" = context1, "old" = context2 (neither is newer)
THRESHOLD = 0.5

# =====================================================================================================
# Decision models: the subject of the test. Each answers the same two yes/no questions with its native
# yes/no type (noul for the System One APIs, predicate for Luna); we read P(yes).
# =====================================================================================================
@dataclass(frozen=True)
class Decider:
    name: str
    model: str          # what goes in the request (pricing key in harness/prices.py)
    api: str            # jev (TypeSafe SDK) | luna (OpenAI /v1/decisions) | systemone (plain HTTP, System One body)
    vendor: str
    released: str
    url: str = ""       # systemone endpoint; {CLOUDFLARE_ACCOUNT_ID} is filled from the environment
    key_env: str = ""   # bearer token env var ("" = no auth: local server)
    send_model: bool = True
    local: bool = False
    revision: str = ""  # local weights: Hugging Face commit
    max_workers: int = 8   # concurrent requests in the throughput run (Perplexity: 10 req/s per organization)
    vendor_latency: str = ""   # the vendor's own latency claim, verbatim in substance, with its source


DECIDERS: tuple[Decider, ...] = (
    Decider("jev", "jev-1.14.0", "jev", "TypeSafe", "2026-10-14",
            vendor_latency="none published (docs.typesafe.ai/models, checked 2026-10-24)"),
    Decider("luna", "gpt-6-luna", "luna", "OpenAI", "2026-10-06 (Decisions API public beta)",
            vendor_latency="none on the docs page; launch coverage quotes 'about 10x faster than the Responses API'"),
    Decider("pplx", "pplx-decider-v1.2", "systemone", "Perplexity", "2026-10-15",
            url="https://api.perplexity.ai/v1/decisions", key_env="PERPLEXITY_API_KEY", max_workers=2,
            vendor_latency="'a few hundred input tokens took under 2 seconds' (quickstart, tests of 2026-09-30, earlier "
                           "model)"),
    Decider("pplx-flash", "pplx-decider-flash-v1.2", "systemone", "Perplexity", "2026-10-15",
            url="https://api.perplexity.ai/v1/decisions", key_env="PERPLEXITY_API_KEY", max_workers=2,
            vendor_latency="none for short inputs (quickstart gives 16 s for ~248k input tokens)"),
    Decider("clef", "clef", "systemone", "Cloudflare", "2026-10-01",
            url="https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/run/@cf/cloudflare/clef",
            key_env="CLOUDFLARE_API_TOKEN",
            vendor_latency="median 209.3 ms, p95 238.6 ms over 43 benchmarks (Cloudflare blog; setup not stated)"),
    Decider("clef-flash", "clef-flash", "systemone", "Cloudflare", "2026-10-01",
            url="https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/run/@cf/cloudflare/clef-flash",
            key_env="CLOUDFLARE_API_TOKEN",
            vendor_latency="median 38.8 ms, p95 122.4 ms over 43 benchmarks (Cloudflare blog; setup not stated)"),
    Decider("strands", "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610", "systemone", "AWS Strands Labs",
            "2026-10-09 (checkpoint; model family 2026-10-01)", url="http://127.0.0.1:8099/v1/systemone",
            send_model=False, local=True, revision="216d1ccb46a9e9b036eb9187b248db98480f77d4",
            vendor_latency="RTX 3090, vendor-measured, not verified here: median 115 ms, p95 299 ms per JevBench "
                           "question (checkpoint v19, evaluation/results.md)"),
)
DECIDER = {d.name: d for d in DECIDERS}
NOT_REACHABLE = {
    "Microsoft-Decision-1": "Azure AI Foundry, preview. The user's deployment failed with 'quota exceeded' in every "
                            "region (2026-10-24); weights are not published. Substitute: pplx-decider-flash-v1.2.",
}
STRANDS_REPO_COMMIT = "b81d443c07438dc5ed51d0e85f9db184ad65d809"   # github.com/strands-labs/strands-decider, 2026-10-23
# The checkpoint is a LoRA adapter (202 MB) + pointer head (13 MB); the server loads the base model at start. Its
# revision comes from the checkpoint's provenance.json (modeling.base_revision), which Strands marks as inferred:
# "Hub main at training time; hosts did not pin it".
STRANDS_BASE = ("Qwen/Qwen3.5-2B-Base", "b1485b2fa6dfa1287294f269f5fb618e03d52d7c")

JOB1_QUESTION = "Do these chunks give different answers to the question?"
JOB2_QUESTION = "Does this answer use the most recent source in the context?"
JOBS = ("job1", "job2")


def job1_state(question: str, context: str) -> str:
    return f"QUESTION:\n{question}\n\nCHUNKS:\n{context}"


def job2_state(question: str, context: str, answer: str) -> str:
    return f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}"


def noul_questions(job: str) -> dict:
    """System One body for Jev, Perplexity, Clef and Strands: one noul, instructions only (no criteria), so the
    wording is exactly the same as Luna's predicate."""
    return {job: {"type": "noul", "instructions": JOB1_QUESTION if job == "job1" else JOB2_QUESTION}}


def luna_questions(job: str) -> list[dict]:
    return [{"type": "predicate", "name": job, "instructions": JOB1_QUESTION if job == "job1" else JOB2_QUESTION}]


# =====================================================================================================
# Generators (answers to check), labelers, faithfulness baseline
# =====================================================================================================
GENERATORS = {"haiku55": "claude-haiku-5-5", "gemma": "google/gemma-4-31B-it"}
LABELERS = {"sonnet": "claude-sonnet-5-5", "sol": "gpt-6.1-sol"}
BASELINE_MODEL = "claude-sonnet-5-5"

# Claude 5.5 rejects temperature; Haiku 5.5 runs with thinking "disabled" (the instruct setting). Gemma 4 31B-it on
# DeepInfra with the model card's standard sampling; thinking is off unless the system prompt starts with <|think|>.
GEN_MAX_TOKENS = 1024
HAIKU_GEN_SETTINGS = {"thinking": {"type": "disabled"}}
DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"
GEMMA_SAMPLING = {"temperature": 1.0, "top_p": 0.95}
GEMMA_EXTRA_BODY = {"top_k": 64}

# pilot (2026-10-25): at 100, Sonnet hit max_tokens on 3 of 40 answers - on hard cases it reasons in prose before
# the JSON despite the format instruction. The parser takes the JSON wherever it is, so only the cap changes.
LABEL_MAX_TOKENS = 400
SOL_LABEL_EFFORT = "low"
SOL_MAX_OUTPUT_TOKENS = 4000
SONNET_SETTINGS = {"thinking": {"type": "between_tools"}}   # Sonnet 5.5's lowest setting; "disabled" returns 400
BASELINE_MAX_TOKENS = 200


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
    """LangChain's format_docs: page contents joined by a blank line."""
    return "\n\n".join(chunks)


# =====================================================================================================
# Default RAG prompt: LangChain Hub rlm/rag-prompt, verbatim (one human message)
# =====================================================================================================
# commit 50442af133e61576e74536c6556cefe1fac147cad032f4377b60c436e6cdcb6e, fetched 2026-10-22 from
# https://api.smith.langchain.com/commits/rlm/rag-prompt/latest
RAG_PROMPT_SOURCE = "LangChain Hub rlm/rag-prompt @ 50442af1 (fetched 2026-10-22)"
RAG_PROMPT = (
    "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep "
    "the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"
)


def gen_prompt(question: str, context: str) -> str:
    return RAG_PROMPT.format(question=question, context=context)


# =====================================================================================================
# Answer labels (two labelers: Claude Sonnet 5.5 and gpt-6.1-sol, same prompt)
# =====================================================================================================
LABELS = ("current", "stale", "blends", "flags", "refuses", "other")
WIKI_LABELS = ("a", "b", "blends", "flags", "refuses", "other")
# Job 2 ground truth: current = yes; stale, blends = no; flags, refuses, other left out. Used only where both
# labelers agree (disagreements are reported, and each labeler alone is a sensitivity check).
JOB2_TRUTH = {"current": 1, "stale": 0, "blends": 0}

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
# Faithfulness baseline: post 1's question ("Is every claim in the answer supported by the context?") as the
# post 2 / post 1 P0 Claude prompt, on Sonnet 5.5. Passage shape = post 1's RAGTruth QA passages.
# =====================================================================================================
def judge_passage(item: dict, context: str) -> dict[str, str]:
    return {"question": item["question"], "passages": context}


__all__ = ["BASELINE_SYSTEM"]
