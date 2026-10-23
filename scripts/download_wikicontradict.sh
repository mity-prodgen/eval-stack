#!/usr/bin/env bash
# Fetches WikiContradict (IBM Research, NeurIPS 2024 Datasets and Benchmarks; MIT license per the dataset card and
# the paper) into data/wikicontradict/ for conflict/'s check set. Pinned to a Hugging Face commit. Not committed -
# data/ is gitignored; conflict/build_items.py draws a fixed-seed sample and commits only the question ids.
set -euo pipefail

DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/wikicontradict"
mkdir -p "$DEST"

REV="c20e361f985ed480a659b35d98b49f2311fcd174"
BASE="https://huggingface.co/datasets/ibm-research/Wikipedia_contradict_benchmark/resolve/$REV"
curl -sfL -o "$DEST/WikiContradict_dataset_v1_rag_qa.csv" "$BASE/WikiContradict_dataset_v1_rag_qa.csv"
curl -sfL -o "$DEST/README.md" "$BASE/README.md"

echo "Downloaded to $DEST:"
ls -l "$DEST"
EXPECTED="cef10054f8aad3e36bca95a0873c2cbb0de8ab4d7b716cadacc5da7aa179ff2b"
GOT="$(shasum -a 256 "$DEST/WikiContradict_dataset_v1_rag_qa.csv" | cut -d' ' -f1)"
[ "$GOT" = "$EXPECTED" ] && echo "sha256 ok" || { echo "sha256 mismatch: $GOT"; exit 1; }
