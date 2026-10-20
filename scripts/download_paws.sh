#!/usr/bin/env bash
# Fetches PAWS-Wiki Labeled (Final), test split, into data/paws/ for semcache/'s external check.
# PAWS: https://github.com/google-research-datasets/paws (data: "may be freely used for any purpose",
# see the repo's LICENSE). The original Google Cloud Storage link returned 403 on 2026-10-19, so this pulls
# the Hugging Face copy at a pinned revision. Not committed - data/ is gitignored; semcache/build_paws.py
# draws a fixed-seed sample and commits only the sampled ids.
set -euo pipefail

DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/paws"
mkdir -p "$DEST"

REV="161ece9501cf0a11f3e48bd356eaa82de46d6a09"
BASE="https://huggingface.co/datasets/google-research-datasets/paws/resolve/$REV"
curl -sfL -o "$DEST/test.parquet" "$BASE/labeled_final/test-00000-of-00001.parquet"
curl -sfL -o "$DEST/LICENSE" "https://raw.githubusercontent.com/google-research-datasets/paws/master/LICENSE"

echo "Downloaded to $DEST:"
ls -l "$DEST"
shasum -a 256 "$DEST/test.parquet"
