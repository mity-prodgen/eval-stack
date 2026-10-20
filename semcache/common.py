"""Paths, seeds, pinned models, thresholds, library defaults and the A3 prompts for post 2 (semantic cache).

`python -m semcache.show_prompts` prints every A3 request body verbatim.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PKG = ROOT / "semcache"
PAIRS_DIR = PKG / "pairs"                   # committed: frozen eval pairs, review CSV, PAWS sample ids
PAIRS = PAIRS_DIR / "pairs.jsonl"
HANDWRITTEN = PAIRS_DIR / "handwritten.jsonl"
PAWS_IDS = PAIRS_DIR / "paws_sample_ids.json"
PAWS_RAW = ROOT / "data" / "paws" / "test.parquet"   # gitignored, scripts/download_paws.sh
RESULTS = PKG / "results"                   # gitignored: scores, rows, logs
FIGURES = PKG / "figures"                   # committed
RESULTS_MD = PKG / "RESULTS.md"
CACHE = ROOT / "cache"                      # shared API response cache (gitignored)

SEED = 20261019
N_PAWS = 300
N_REVIEW = 30

# ------------------------------------------------------------------------------- embedding models
@dataclass(frozen=True)
class Embedder:
    name: str            # short name used in tables
    provider: str        # openai | cohere | voyage | gemini | local
    model_id: str        # exactly what goes in the request (or the HF repo)
    revision: str = ""   # HF commit for local models
    dims: int | None = None
    settings: str = ""   # request settings, for the README table


EMBEDDERS: tuple[Embedder, ...] = (
    Embedder("oai-3-small", "openai", "text-embedding-3-small", dims=1536, settings="default dimensions"),
    Embedder("oai-3-large", "openai", "text-embedding-3-large", dims=3072, settings="default dimensions"),
    Embedder("cohere-v4", "cohere", "embed-v4.0", dims=1536,
             settings='input_type="search_query" on both sides (a cache compares query to query), float'),
    Embedder("voyage-4", "voyage", "voyage-4", dims=1024, settings="input_type=None (symmetric), float"),
    Embedder("gemini-2", "gemini", "gemini-embedding-2", dims=3072,
             settings='prefix "task: sentence similarity | query: " (the documented similarity prompt)'),
    Embedder("minilm", "local", "sentence-transformers/all-MiniLM-L6-v2",
             revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41", dims=384, settings="sentence-transformers, normalized"),
    Embedder("bge-m3", "local", "BAAI/bge-m3", revision="5617a9f61b028005a4858fdac845db406aefb181", dims=1024,
             settings="dense vector only, sentence-transformers, normalized"),
    Embedder("langcache-v1", "local", "redis/langcache-embed-v1", revision="2f6ffda359ab94a9c222947bfd87ea8f85ea7e0a",
             dims=768, settings="sentence-transformers, normalized (RedisVL SemanticCache default vectorizer)"),
)
EMBEDDER = {e.name: e for e in EMBEDDERS}
GEMINI_SIMILARITY_PREFIX = "task: sentence similarity | query: "

# ------------------------------------------------------------------------------- thresholds
# Task asks for 0.80-0.99. Several library defaults sit below 0.80 in cosine terms (see DEFAULTS), and the
# sweep is free once embeddings exist, so it starts at 0.50; charts focus on 0.80-0.99 and mark the defaults.
THRESHOLDS = tuple(round(0.50 + 0.01 * i, 2) for i in range(50))   # 0.50 .. 0.99
TASK_THRESHOLDS = tuple(t for t in THRESHOLDS if t >= 0.80)


@dataclass(frozen=True)
class LibDefault:
    library: str
    setting: str             # parameter = default, as in the source
    cosine: float | None     # equivalent cosine-similarity threshold for unit vectors, if derivable
    derivation: str
    source: str              # file in the installed package, version, checked date


DEFAULTS_CHECKED = "2026-10-19"
DEFAULTS: tuple[LibDefault, ...] = (
    LibDefault("RedisVL SemanticCache (0.28.0)", "distance_threshold=0.1, vectorizer redis/langcache-embed-v1",
               0.90, "Redis COSINE distance = 1 - cos; hit if distance <= 0.1",
               "redisvl/extensions/cache/llm/semantic.py (redisvl 0.28.0)"),
    LibDefault("langchain-redis RedisSemanticCache (0.2.6)", "distance_threshold=0.2", 0.80,
               "passed to RedisVL SemanticCache (COSINE distance)", "langchain_redis/cache.py (langchain-redis 0.2.6)"),
    LibDefault("langchain-community RedisSemanticCache (0.4.2)", "score_threshold=0.2", 0.80,
               "passed as distance_threshold to RedisVL (COSINE distance)", "langchain_community/cache.py (0.4.2)"),
    LibDefault("langchain-community CassandraSemanticCache (0.4.2)", 'score_threshold=0.85, similarity_measure="dot"',
               0.70, "Cassandra similarity_dot_product returns (1 + dot)/2 (fits the CQL docs' worked example); "
               "(1 + cos)/2 >= 0.85 -> cos >= 0.70", "langchain_community/cache.py (0.4.2); "
               "cassandra.apache.org/doc/latest/cassandra/developing/cql/functions.html"),
    LibDefault("GPTCache (0.1.44)", "similarity_threshold=0.8 with SearchDistanceEvaluation(max_distance=4.0) "
               "+ FAISS L2", 0.60, "score = 4 - L2^2, hit if score >= 0.8 x 4 = 3.2 -> L2^2 <= 0.8; for unit "
               "vectors L2^2 = 2 - 2cos -> cos >= 0.60. Its own default ONNX embedding is not normalized, so this "
               "is an equivalence for unit vectors only. Note cache.init()'s bare default is exact match.",
               "gptcache/config.py, gptcache/adapter/adapter.py, similarity_evaluation/distance.py (0.1.44)"),
    LibDefault("langchain-community OpenSearchSemanticCache (0.4.2)", "score_threshold=0.2, space_type l2", None,
               "OpenSearch l2 score = 1/(1 + L2^2) >= 0.2 holds for every L2^2 <= 4, i.e. any pair of unit vectors: "
               "effectively no threshold (derived from source, not run against OpenSearch)",
               "langchain_community/cache.py, vectorstores/opensearch_vector_search.py (0.4.2)"),
    LibDefault("langchain-community SingleStoreDBSemanticCache (0.4.2)", "search_threshold=0.2, DOT_PRODUCT", 0.20,
               "minimum dot product; = cosine for unit vectors", "langchain_community/cache.py (0.4.2)"),
)
DEFAULT_COSINE = 0.90   # "the default threshold" for A0/A2/A3: RedisVL's, the library whose default model we test

# ------------------------------------------------------------------------------- A1, A3
A1_KEEP = 0.80          # A1: highest threshold that keeps >= 80% of A0's control hits (chosen out of fold)
A3_PASS = 0.5           # serve if P(yes) >= 0.5

JEV_MODEL = "jev-1.13.0"
LUNA_MODEL = "gpt-6-luna"
DECIDERS = ("jev", "luna")
DECIDER_MODELS = {"jev": JEV_MODEL, "luna": LUNA_MODEL}

# Kept generic on purpose: no list of slot types, so the deciders are not primed toward the traps we wrote.
A3_QUESTION = "Do these two queries need the same answer?"
JEV_A3 = {
    "same_answer": {
        "type": "noul",
        "instructions": "`cached_query` and `new_query` need the same answer: an answer written for `cached_query` "
                        "is a correct and complete answer to `new_query`.",
        "criteria": {
            "true": "One answer serves both: the queries ask for the same thing about the same specifics.",
            "false": "The queries differ in something the answer depends on, so one answer cannot serve both.",
        },
    }
}
LUNA_A3_QUESTION = (
    A3_QUESTION + " True only if an answer written for the CACHED QUERY is a correct and complete answer to the NEW "
    "QUERY. False if they differ in anything the answer depends on."
)


def jev_state(cached: str, new: str) -> dict[str, str]:
    return {"cached_query": cached, "new_query": new}


def luna_input(cached: str, new: str) -> str:
    return f"CACHED QUERY:\n{cached}\n\nNEW QUERY:\n{new}"


def luna_questions() -> list[dict]:
    return [{"type": "predicate", "name": "same_answer", "instructions": LUNA_A3_QUESTION}]
