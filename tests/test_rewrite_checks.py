from rewrite import added_claims, check, span_scores

ITEM = {
    "response": "The restaurant opened in 1998 and one reviewer did appreciate the portion size.",
    "labels": [{"text": "one reviewer did appreciate the portion size"}],
}


def test_span_survives_verbatim_and_restyled_context():
    rw = "Opened in 1998, this restaurant wins praise: one reviewer did appreciate the portion size."
    ok, info = check(ITEM, rw)
    assert ok, info
    assert min(info["span_scores"]) >= 90


def test_span_tolerates_quote_and_whitespace_noise():
    assert span_scores(["it’s   open late"], "Without question it's open late.")[0] == 100


def test_lost_span_fails():
    ok, info = check(ITEM, "Opened in 1998, the restaurant's portions drew a mixed response.")
    assert not ok and "span_lost" in info["reasons"]


def test_added_correctness_claim_fails_but_preexisting_word_is_allowed():
    ok, info = check(ITEM, "Verified: opened in 1998, and one reviewer did appreciate the portion size.")
    assert not ok and info["added_claims"]
    assert added_claims("The date was confirmed as correct.", "The date was confirmed as correct.") == []


def test_clean_item_without_spans_passes_on_style_change():
    ok, _ = check({"response": "It opened in 1998.", "labels": []}, "It opened its doors in 1998.")
    assert ok


def test_empty_rewrite_fails():
    ok, info = check({"response": "x", "labels": []}, "  ")
    assert not ok and "empty" in info["reasons"]
