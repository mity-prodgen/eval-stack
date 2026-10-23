import re
import subprocess

from common import ROOT

KEY_PATTERNS = [r"sk-ant-[A-Za-z0-9_-]{20,}", r"sk-proj-[A-Za-z0-9_-]{20,}", r"sk-[A-Za-z0-9]{32,}"]


def test_env_and_data_are_gitignored():
    ignored = (ROOT / ".gitignore").read_text().split()
    for entry in (".env", "/data/", "cache/", "persuasion/results/", "semcache/results/", "conflict/results/"):
        assert entry in ignored


def test_no_key_like_strings_in_tracked_or_staged_files():
    files = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT,
                           capture_output=True, text=True, check=True).stdout.split()
    for f in files:
        p = ROOT / f
        if not p.is_file() or p.suffix in (".png",):
            continue
        text = p.read_text(errors="ignore")
        for pat in KEY_PATTERNS:
            assert not re.search(pat, text), f"key-like string in {f}"
