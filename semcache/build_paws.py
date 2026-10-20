#!/usr/bin/env python3
"""Fixed-seed PAWS sample (external check): 300 pairs from PAWS-Wiki Labeled (Final), test split.

PAWS pairs come from word swapping and back translation, so most non-paraphrases share almost every word with
their partner. label 1 (paraphrase) -> "same", label 0 -> "different". The sample is stratified 150/150 so
both rates have the same n. Only the sampled row ids are committed (pairs/paws_sample_ids.json); the text is
rebuilt from data/paws/test.parquet (scripts/download_paws.sh).

    python -m semcache.build_paws
"""

from __future__ import annotations

import json
import random

from .common import N_PAWS, PAWS_IDS, PAWS_RAW, RESULTS, SEED


def load_raw() -> list[dict]:
    import pyarrow.parquet as pq
    if not PAWS_RAW.exists():
        raise FileNotFoundError(f"{PAWS_RAW} missing - run scripts/download_paws.sh")
    return pq.read_table(PAWS_RAW).to_pylist()


def sample(raw: list[dict]) -> list[int]:
    rng = random.Random(SEED + 2)
    out = []
    for label in (0, 1):
        ids = sorted(r["id"] for r in raw if r["label"] == label)
        out += rng.sample(ids, N_PAWS // 2)
    return sorted(out)


def load_paws() -> list[dict]:
    """The sampled pairs in the same row shape as pairs.jsonl."""
    ids = json.loads(PAWS_IDS.read_text())["ids"]
    raw = {r["id"]: r for r in load_raw()}
    return [{"id": f"paws{i}", "cached": raw[i]["sentence1"], "new": raw[i]["sentence2"],
             "label": "same" if raw[i]["label"] == 1 else "different", "slot_type": "paws", "domain": "wikipedia",
             "source": "paws"} for i in ids]


def main() -> None:
    raw = load_raw()
    ids = sample(raw)
    if PAWS_IDS.exists():
        frozen = json.loads(PAWS_IDS.read_text())["ids"]
        print(f"{PAWS_IDS} is frozen; a rebuild {'matches it' if frozen == ids else 'DIFFERS - not overwritten'}")
    else:
        PAWS_IDS.write_text(json.dumps({"source": "google-research-datasets/paws labeled_final test "
                                        "(HF revision 161ece9501cf0a11f3e48bd356eaa82de46d6a09)",
                                        "seed": SEED + 2, "n": len(ids), "ids": ids}, indent=1))
    rows = load_paws()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "paws_sample.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} PAWS pairs: {sum(r['label'] == 'same' for r in rows)} same / "
          f"{sum(r['label'] == 'different' for r in rows)} different (of {len(raw)} in the test split)")


if __name__ == "__main__":
    main()
