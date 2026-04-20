from __future__ import annotations

import gzip
import re
import urllib.request
from pathlib import Path
from typing import Iterable

import regex

WORD_RE = regex.compile(r"\w+", flags=regex.UNICODE)
SENT_RE = re.compile(r"[.!?]+\s+")

MODEL_CACHE_DIR = Path.home() / ".cache" / "t0_training" / "filter_models"
CURSED_BANLIST_URL = (
    "https://raw.githubusercontent.com/allenai/datamap-rs/main/"
    "banlists/madlad400_cursed.txt.gz"
)
CURSED_BANLIST_PATH = MODEL_CACHE_DIR / "madlad400_cursed.txt.gz"


def ensure_cursed_banlist() -> Path | None:
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if CURSED_BANLIST_PATH.exists():
        return CURSED_BANLIST_PATH
    try:
        urllib.request.urlretrieve(CURSED_BANLIST_URL, CURSED_BANLIST_PATH)
        return CURSED_BANLIST_PATH
    except Exception:
        return None


def split_sentences(text: str) -> list[str]:
    return [s for s in SENT_RE.split(text) if s.strip()]


def list_case_rule(sentence: str) -> bool:
    words = WORD_RE.findall(sentence)
    if len(words) < 12:
        return False
    starts_upper = sum(1 for w in words if w and w[0].isupper())
    return (starts_upper / len(words)) > 0.5


def _load_cursed_rules(path: Path) -> tuple[list[str], list[re.Pattern[str]]]:
    if not path.exists():
        return [], []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    if len(lines) < 4:
        literals = lines
        regexes: list[re.Pattern[str]] = []
    else:
        literals = lines[:-4]
        regexes = [re.compile(p) for p in lines[-4:]]
    return literals, regexes


def cursed_rule(sentence: str, literals: Iterable[str], regexes: Iterable[re.Pattern[str]]) -> bool:
    for lit in literals:
        if lit in sentence:
            return True
    for rx in regexes:
        if rx.search(sentence):
            return True
    return False


def madlad400_filter(
    text: str,
    cursed_banlist_path: Path | None = None,
    threshold: float = 0.2,
    remove_too_short: bool = True,
) -> dict[str, object]:
    sentences = split_sentences(text)
    if len(sentences) < 5:
        return {
            "passed": not remove_too_short,
            "reason": "killed:too_short" if remove_too_short else "too_short_but_allowed",
            "suspicious_ratio": 1.0,
            "suspicious_sentences": len(sentences),
            "sentence_count": len(sentences),
        }

    literals: list[str] = []
    regexes: list[re.Pattern[str]] = []
    if cursed_banlist_path is not None:
        literals, regexes = _load_cursed_rules(cursed_banlist_path)

    suspicious = 0
    for s in sentences:
        flagged = list_case_rule(s) or cursed_rule(s, literals, regexes)
        if flagged:
            suspicious += 1

    ratio = suspicious / len(sentences)
    passed = ratio < threshold
    return {
        "passed": passed,
        "reason": None if passed else f"suspicious_ratio={ratio:.3f}",
        "suspicious_ratio": ratio,
        "suspicious_sentences": suspicious,
        "sentence_count": len(sentences),
    }
