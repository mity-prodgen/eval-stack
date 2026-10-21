"""Embedding clients (hosted + local), cached per text, and cosine scores per pair.

Every vector goes through harness.cache.DiskCache (namespace "semcache"), one record per (model, settings,
text), so reruns and re-analysis cost nothing. Vectors are stored as base64 float32 to keep the cache small.
Hosted calls are batched over the uncached texts only; each batch's billed usage is appended to
results/embed_usage.jsonl.

Pitfall handled here: gemini-embedding-2 returns ONE aggregated embedding when `contents` is a list of strings,
so each text is wrapped in its own Content, and every client asserts one vector per input text.
"""

from __future__ import annotations

import base64
import json
import time
from typing import Callable, Iterable

import numpy as np

from harness.cache import DiskCache, cache_key
from harness.prices import usage_cost

from .common import CACHE, GEMINI_SIMILARITY_PREFIX, RESULTS, Embedder

BATCH = {"openai": 256, "cohere": 96, "voyage": 128, "gemini": 50, "local": 64}
SETTINGS_VERSION = 1  # bump if a request setting changes, so old vectors are not reused


def text_key(e: Embedder, text: str) -> str:
    return cache_key("embed", SETTINGS_VERSION, e.provider, e.model_id, e.revision, e.settings, text)


def _pack(v: np.ndarray) -> str:
    return base64.b64encode(np.asarray(v, dtype=np.float32).tobytes()).decode()


def _unpack(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype=np.float32)


# ----------------------------------------------------------------------------------- providers
_CLIENTS: dict[str, object] = {}


def _openai(e: Embedder, texts: list[str]) -> tuple[list[list[float]], int | None, dict]:
    from openai import OpenAI
    r = OpenAI(max_retries=4, timeout=60.0).embeddings.create(model=e.model_id, input=texts)
    vecs = [d.embedding for d in sorted(r.data, key=lambda d: d.index)]
    return vecs, r.usage.prompt_tokens, {"model": r.model}


def _cohere(e: Embedder, texts: list[str]) -> tuple[list[list[float]], int | None, dict]:
    import cohere
    r = cohere.ClientV2(timeout=60).embed(texts=texts, model=e.model_id, input_type="search_query",
                                          embedding_types=["float"])
    bu = getattr(r.meta, "billed_units", None)
    return list(r.embeddings.float_), getattr(bu, "input_tokens", None), {"id": r.id}


def _voyage(e: Embedder, texts: list[str]) -> tuple[list[list[float]], int | None, dict]:
    import voyageai
    r = voyageai.Client(max_retries=4, timeout=60).embed(texts, model=e.model_id, input_type=None)
    return r.embeddings, r.total_tokens, {}


def _gemini(e: Embedder, texts: list[str]) -> tuple[list[list[float]], int | None, dict]:
    from google import genai
    from google.genai import types
    contents = [types.Content(parts=[types.Part(text=GEMINI_SIMILARITY_PREFIX + t)]) for t in texts]
    # keep the client referenced: an inline genai.Client() is garbage-collected mid-call, which closes its
    # HTTP client ("Cannot send a request, as the client has been closed")
    client = _CLIENTS.setdefault("gemini", genai.Client())
    r = client.models.embed_content(model=e.model_id, contents=contents)
    # the response carries no token count; cost is estimated from the text (see estimate.py)
    return [x.values for x in r.embeddings], None, {}


_LOCAL: dict[str, object] = {}


def _local(e: Embedder, texts: list[str]) -> tuple[list[list[float]], int | None, dict]:
    from sentence_transformers import SentenceTransformer
    m = _LOCAL.get(e.model_id)
    if m is None:
        m = _LOCAL[e.model_id] = SentenceTransformer(e.model_id, revision=e.revision)
    vecs = m.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    return [v for v in vecs], None, {}


PROVIDERS: dict[str, Callable] = {"openai": _openai, "cohere": _cohere, "voyage": _voyage, "gemini": _gemini,
                                  "local": _local}


# ----------------------------------------------------------------------------------- cached embedding
def uncached(e: Embedder, texts: Iterable[str], cache: DiskCache) -> list[str]:
    return [t for t in dict.fromkeys(texts) if cache.get(text_key(e, t)) is None]


def embed_texts(e: Embedder, texts: list[str], cache: DiskCache, log: Callable[[str], None] = print,
                call: Callable | None = None) -> dict[str, np.ndarray]:
    """Vectors for every text (unit-normalized), calling the API only for uncached ones."""
    call = call or PROVIDERS[e.provider]
    todo = uncached(e, texts, cache)
    RESULTS.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(todo), BATCH[e.provider]):
        chunk = todo[i: i + BATCH[e.provider]]
        t0 = time.monotonic()
        vecs, billed, meta = call(e, chunk)
        if len(vecs) != len(chunk):
            raise RuntimeError(f"{e.name}: {len(vecs)} vectors for {len(chunk)} texts - refusing to cache")
        ms = (time.monotonic() - t0) * 1000
        for t, v in zip(chunk, vecs):
            v = np.asarray(v, dtype=np.float32)
            if e.dims and v.shape[0] != e.dims:
                raise RuntimeError(f"{e.name}: got {v.shape[0]} dims, expected {e.dims}")
            cache.set(text_key(e, t), {"ok": True, "model": e.model_id, "dims": int(v.shape[0]),
                                       "norm": float(np.linalg.norm(v)), "vec": _pack(v), "cached_at": time.time()})
        if e.provider != "local":
            cost = usage_cost(e.model_id, billed) if billed is not None else None
            with (RESULTS / "embed_usage.jsonl").open("a") as f:
                f.write(json.dumps({"embedder": e.name, "model": e.model_id, "n_texts": len(chunk),
                                    "billed_input_tokens": billed, "cost_usd": cost, "latency_ms": ms,
                                    "ts": time.time(), **meta}) + "\n")
        log(f"  [{e.name}] embedded {min(i + len(chunk), len(todo))}/{len(todo)}")
    out = {}
    for t in dict.fromkeys(texts):
        v = _unpack(cache.get(text_key(e, t))["vec"])
        out[t] = v / np.linalg.norm(v)
    return out


def cosine_scores(e: Embedder, pairs: list[dict], cache: DiskCache, **kw) -> dict[str, float]:
    texts = [t for p in pairs for t in (p["cached"], p["new"])]
    vec = embed_texts(e, texts, cache, **kw)
    return {p["id"]: float(np.dot(vec[p["cached"]], vec[p["new"]])) for p in pairs}


def semcache_cache() -> DiskCache:
    return DiskCache(CACHE, "semcache")
