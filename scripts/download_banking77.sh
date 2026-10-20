#!/usr/bin/env bash
# Fetches BANKING77 (test split) into data/banking77/ for semcache/'s real-query seeds.
# BANKING77 (Casanueva et al. 2020, PolyAI), CC BY 4.0: https://github.com/PolyAI-LDN/task-specific-datasets
# Pinned to a commit. Not committed - data/ is gitignored; semcache/generate_pairs.py records the sampled rows.
set -euo pipefail

DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/banking77"
mkdir -p "$DEST"

REV="57ec275d8078af65b7731c2a98be812d844a6d6b"
curl -sfL -o "$DEST/test.csv" "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/$REV/banking_data/test.csv"

echo "Downloaded to $DEST:"
wc -l "$DEST/test.csv"
shasum -a 256 "$DEST/test.csv"
