import json
from collections import Counter

import pytest

from build_sample import draw, load_ragtruth, post2_sample
from common import RAGTRUTH, ROOT

needs_data = pytest.mark.skipif(not (RAGTRUTH / "response.jsonl").exists(), reason="run scripts/download_ragtruth.sh")


@pytest.fixture(scope="module")
def p2():
    return post2_sample(*load_ragtruth())


@needs_data
def test_sample_is_stratified_seeded_and_inside_post2(p2):
    s = draw(p2)
    assert len(s) == 120
    cells = Counter((c["task_type"], c["hallucinated"]) for c in s)
    assert set(cells.values()) == {20} and len(cells) == 6
    assert all(len(c["labels"]) >= 1 for c in s if c["hallucinated"])
    assert all(len(c["labels"]) == 0 for c in s if not c["hallucinated"])
    assert {c["response_id"] for c in s} <= {c["response_id"] for c in p2}
    assert [c["response_id"] for c in draw(p2)] == [c["response_id"] for c in s]


@needs_data
def test_committed_ids_match_rebuild(p2):
    ids = json.loads((ROOT / "persuasion" / "sample_ids.json").read_text())
    assert [i["response_id"] for i in ids["items"]] == [c["response_id"] for c in draw(p2)]
    assert ids["overlap_with_post2_sample"] == {"n": 120, "of": 120}


@needs_data
def test_annotated_spans_are_verbatim_in_responses(p2):
    for c in draw(p2):
        for l in c["labels"]:
            assert c["response"][l["start"]:l["end"]] == l["text"] or l["text"] in c["response"]
