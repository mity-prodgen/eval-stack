"""semcache: frozen data, request shapes, parsing, embedding cache, budget gate, library defaults. No network."""

import hashlib
import json
import sys

import numpy as np
import pytest

from semcache import build_pairs, decide, embed, estimate, run
from semcache.common import (DEFAULTS, EMBEDDER, EMBEDDERS, HANDWRITTEN, PAIRS, PAIRS_DIR, PAWS_IDS, THRESHOLDS,
                             Embedder)
from harness.cache import DiskCache


# ---------------------------------------------------------------- data
def test_frozen_pairs_match_manifest_and_rebuild():
    manifest = json.loads((PAIRS_DIR / "MANIFEST.json").read_text())
    assert hashlib.sha256(PAIRS.read_bytes()).hexdigest() == manifest["pairs.jsonl"]
    assert hashlib.sha256(HANDWRITTEN.read_bytes()).hexdigest() == manifest["handwritten.jsonl"]
    rows, _ = build_pairs.build()
    assert "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows) == PAIRS.read_text()


def test_pairs_shape_and_balance():
    rows = [json.loads(l) for l in PAIRS.open()]
    assert len({r["id"] for r in rows}) == len(rows)
    traps = [r for r in rows if r["label"] == "different"]
    controls = [r for r in rows if r["label"] == "same"]
    assert 190 <= len(traps) <= 210 and 190 <= len(controls) <= 210
    counts = {s: sum(r["slot_type"] == s for r in traps) for s in build_pairs.SLOT_TYPES}
    assert max(counts.values()) - min(counts.values()) <= 3, counts
    assert {r["domain"] for r in rows} == {"airline", "banking", "dev-tools", "support"}
    assert all(r["cached"].strip().lower() != r["new"].strip().lower() for r in rows)


def test_review_csv_has_30_rows():
    lines = (PAIRS_DIR / "review.csv").read_text().strip().splitlines()
    assert len(lines) == 31


def test_paws_ids_frozen_and_balanced():
    d = json.loads(PAWS_IDS.read_text())
    assert d["n"] == len(d["ids"]) == 300 and len(set(d["ids"])) == 300


def test_smoke_pairs_cover_every_slot_type_and_are_stable():
    pairs = run.load_pairs()
    s = run.smoke_pairs(pairs)
    assert len(s) == 20 and sum(p["label"] == "different" for p in s) == 10
    assert {p["slot_type"] for p in s if p["label"] == "different"} == set(build_pairs.SLOT_TYPES)
    assert [p["id"] for p in s] == [p["id"] for p in run.smoke_pairs(pairs)]


def test_threshold_grid_covers_task_range():
    assert THRESHOLDS[0] == 0.5 and THRESHOLDS[-1] == 0.99 and 0.8 in THRESHOLDS and len(THRESHOLDS) == 50


# ---------------------------------------------------------------- A3 requests and parsing
PAIR = {"id": "x", "cached": "Fare from DXB to LHR", "new": "Fare from LHR to DXB"}


def test_a3_request_shapes():
    jev, jev114, luna = decide.plan([PAIR], ["jev113", "jev114", "luna"])
    assert jev.model == "jev-1.13.0" and jev114.model == "jev-1.14.0"
    assert jev.payload == jev114.payload and jev.key != jev114.key
    assert jev.payload["state"] == {"cached_query": PAIR["cached"], "new_query": PAIR["new"]}
    assert jev.payload["questions"]["same_answer"]["type"] == "noul"
    assert luna.payload["model"] == "gpt-6-luna" and luna.payload["questions"][0]["type"] == "predicate"
    assert luna.payload["input"] == "CACHED QUERY:\nFare from DXB to LHR\n\nNEW QUERY:\nFare from LHR to DXB"
    assert jev.key != luna.key and len(jev.key) == 32


def test_a3_parsers():
    ok = decide.parse_jev({"model": "jev-1.14.0", "answers": {"same_answer": {"noul": 0.2}}, "usage": {"input_tokens": 100}},
                          "jev-1.14.0")
    assert ok["status"] == "ok" and ok["p_yes"] == 0.2 and ok["cost_usd"] > 0
    assert decide.parse_jev({"answers": {}}, "jev-1.13.0")["status"] == "parse_error"
    luna = decide.parse_luna({"answers": [{"name": "same_answer", "type": "predicate", "probability": 0.9}],
                              "usage": {"input_tokens": 50}})
    assert luna["status"] == "ok" and luna["p_yes"] == 0.9
    assert decide.parse_luna({"answers": [{"name": "same_answer", "type": "refusal"}]})["status"] == "refusal"


# ---------------------------------------------------------------- embedding cache
def test_embed_cache_calls_only_uncached_and_rejects_count_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(embed, "RESULTS", tmp_path)
    cache = DiskCache(tmp_path, "semcache")
    e = Embedder("fake", "openai", "text-embedding-3-small", dims=3)
    seen = []

    def call(_, texts):
        seen.append(list(texts))
        return [[1.0, float(i), 0.0] for i, _ in enumerate(texts)], 7, {}

    a = embed.embed_texts(e, ["a", "b"], cache, log=lambda _: None, call=call)
    embed.embed_texts(e, ["a", "b", "c"], cache, log=lambda _: None, call=call)
    assert seen == [["a", "b"], ["c"]]
    assert np.isclose(np.linalg.norm(a["b"]), 1.0)
    usage = [json.loads(l) for l in (tmp_path / "embed_usage.jsonl").open()]
    assert usage[0]["billed_input_tokens"] == 7 and usage[0]["cost_usd"] > 0
    with pytest.raises(RuntimeError, match="vectors for"):
        embed.embed_texts(e, ["d", "e"], cache, log=lambda _: None, call=lambda _, t: ([[1.0, 0.0, 0.0]], 1, {}))
    assert cache.get(embed.text_key(e, "d")) is None


def test_cosine_scores(tmp_path, monkeypatch):
    monkeypatch.setattr(embed, "RESULTS", tmp_path)
    vecs = {"x": [1.0, 0.0], "y": [1.0, 1.0]}
    e = Embedder("fake", "local", "fake/model", dims=2)
    s = embed.cosine_scores(e, [{"id": "p", "cached": "x", "new": "y"}], DiskCache(tmp_path, "s"),
                            log=lambda _: None, call=lambda _, t: ([vecs[i] for i in t], None, {}))
    assert s["p"] == pytest.approx(2 ** -0.5, abs=1e-6)


def test_gemini_wraps_each_text_in_its_own_content(monkeypatch):
    from google import genai
    from google.genai import types
    captured = {}

    class FakeModels:
        def embed_content(self, model, contents, config=None):
            captured["contents"] = contents
            return types.EmbedContentResponse(embeddings=[types.ContentEmbedding(values=[1.0, 0.0]) for _ in contents])

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(genai, "Client", lambda *a, **k: FakeClient())
    vecs, _, _ = embed._gemini(EMBEDDER["gemini-2"], ["one", "two"])
    assert len(vecs) == 2 and all(isinstance(c, types.Content) for c in captured["contents"])
    assert captured["contents"][0].parts[0].text.startswith("task: sentence similarity | query: ")


# ---------------------------------------------------------------- estimates and gate
def test_estimates():
    texts = ["How do I reset my password?"] * 3
    assert estimate.estimate_embed(EMBEDDER["minilm"], texts).usd == 0
    for e in EMBEDDERS:
        if e.provider != "local":
            assert estimate.estimate_embed(e, texts).usd > 0
    est = estimate.estimate_decisions(decide.plan([PAIR], ["jev113", "jev114", "luna"]))
    assert est["jev113"].usd > 0 and est["jev114"].usd > 0 and est["luna"].usd > 0


def test_live_run_over_cap_aborts_before_any_call(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(run, "cosine_scores", lambda *a, **k: called.append(1))
    monkeypatch.setattr(run, "run_decisions", lambda *a, **k: called.append(1))
    monkeypatch.setattr(run, "semcache_cache", lambda: DiskCache(tmp_path, "semcache"))
    monkeypatch.setattr(sys, "argv", ["run", "--live", "--max-usd", "0.00001", "--no-paws"])
    with pytest.raises(estimate.BudgetExceeded):
        run.main()
    assert called == []


# ---------------------------------------------------------------- library defaults match installed source
def test_library_defaults_match_installed_source():
    import inspect
    rv = pytest.importorskip("redisvl.extensions.cache.llm.semantic")
    assert inspect.signature(rv.SemanticCache.__init__).parameters["distance_threshold"].default == 0.1
    assert 'model="redis/langcache-embed-v1"' in inspect.getsource(rv.SemanticCache.__init__)
    lr = pytest.importorskip("langchain_redis.cache")
    assert inspect.signature(lr.RedisSemanticCache.__init__).parameters["distance_threshold"].default == 0.2
    lc = pytest.importorskip("langchain_community.cache")
    assert inspect.signature(lc.RedisSemanticCache.__init__).parameters["score_threshold"].default == 0.2
    assert lc.CASSANDRA_SEMANTIC_CACHE_DEFAULT_SCORE_THRESHOLD == 0.85
    assert lc.CASSANDRA_SEMANTIC_CACHE_DEFAULT_DISTANCE_METRIC == "dot"
    assert inspect.signature(lc.OpenSearchSemanticCache.__init__).parameters["score_threshold"].default == 0.2
    assert inspect.signature(lc.SingleStoreDBSemanticCache.__init__).parameters["search_threshold"].default == 0.2
    gc = pytest.importorskip("gptcache.config")
    assert inspect.signature(gc.Config.__init__).parameters["similarity_threshold"].default == 0.8
    from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
    assert inspect.signature(SearchDistanceEvaluation.__init__).parameters["max_distance"].default == 4.0
    assert {d.cosine for d in DEFAULTS} == {0.9, 0.8, 0.7, 0.6, None, 0.2}
