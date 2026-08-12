from __future__ import annotations

from collections import Counter

import regex

WORD_RE = regex.compile(r"\w+", flags=regex.UNICODE)


def _best_ngram_windows(elements: list[str], ngram_size: int) -> tuple[set[int], int]:
    if len(elements) < ngram_size:
        return set(), 0

    windows: list[tuple[str, ...]] = []
    window_len: list[int] = []
    for i in range(len(elements) - ngram_size + 1):
        w = tuple(elements[i : i + ngram_size])
        windows.append(w)
        window_len.append(sum(len(x) for x in w))

    counts = Counter(windows)
    repeated = {k: v for k, v in counts.items() if v > 1}
    if not repeated:
        return set(), 0

    if ngram_size <= 4:
        max_count = max(repeated.values())
        candidates = [k for k, c in repeated.items() if c == max_count]
        best = max(candidates, key=lambda ng: sum(len(x) for x in ng))
        repeated = {best: repeated[best]}

    covered: set[int] = set()
    covered_chars = 0
    for idx, w in enumerate(windows):
        if w in repeated:
            for j in range(idx, idx + ngram_size):
                if j not in covered:
                    covered.add(j)
                    covered_chars += len(elements[j])
    return covered, covered_chars


def rep_counter_fraction(elements: list[str], ngram_size: int, weighted: bool) -> float:
    if ngram_size == 1 and len(elements) == 0:
        return 1.0
    if len(elements) == 0 or len(elements) < ngram_size:
        return 0.0

    if ngram_size == 1:
        counts = Counter(elements)
        repeated = {k for k, v in counts.items() if v > 1}
        if not repeated:
            return 0.0
        if weighted:
            total = sum(len(e) for e in elements)
            covered = sum(len(e) for e in elements if e in repeated)
            return 0.0 if total == 0 else (covered / total)
        return sum(1 for e in elements if e in repeated) / len(elements)

    covered_indices, covered_chars = _best_ngram_windows(elements, ngram_size)
    if weighted:
        total_chars = sum(len(e) for e in elements)
        return 0.0 if total_chars == 0 else (covered_chars / total_chars)
    return len(covered_indices) / len(elements)


RULES = [
    ("lines", 1, False, 0.30),
    ("paragraphs", 1, False, 0.30),
    ("lines", 1, True, 0.20),
    ("paragraphs", 1, True, 0.20),
    ("words", 2, True, 0.20),
    ("words", 3, True, 0.18),
    ("words", 4, True, 0.16),
    ("words", 5, True, 0.15),
    ("words", 6, True, 0.14),
    ("words", 7, True, 0.13),
    ("words", 8, True, 0.12),
    ("words", 9, True, 0.11),
    ("words", 10, True, 0.10),
]


def _elements(text: str, kind: str) -> list[str]:
    if kind == "lines":
        return [x for x in text.split("\n") if x]
    if kind == "paragraphs":
        return [x for x in text.split("\n\n") if x]
    if kind == "words":
        return [x for x in WORD_RE.findall(text) if x]
    raise ValueError(f"invalid kind: {kind}")


def massive_web_repetition_filter(text: str) -> dict[str, object]:
    failures: list[tuple[str, float, float]] = []
    max_fraction = 0.0
    max_rule = ""

    for kind, n, weighted, threshold in RULES:
        elems = _elements(text, kind)
        frac = rep_counter_fraction(elems, n, weighted)
        if frac > max_fraction:
            max_fraction = frac
            max_rule = f"{kind} n={n} {'weighted' if weighted else 'unweighted'}"
        if frac > threshold:
            failures.append((f"{kind}:{n}:{'w' if weighted else 'u'}", frac, threshold))

    return {
        "passed": len(failures) == 0,
        "max_fraction": max_fraction,
        "max_rule": max_rule,
        "failures": failures,
    }
