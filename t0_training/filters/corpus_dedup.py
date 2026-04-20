from __future__ import annotations

import json
import pickle
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import xxhash

SUBSTRING_DEDUP_MIN_BYTES = 500
SUBSTRING_DEDUP_MIN_FRACTION = 0.05


def exact_hash_128(text: str) -> bytes:
    return xxhash.xxh3_128(text.encode("utf-8")).digest()


def save_exact_hashes(hashes: set[bytes], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(hashes, f)


def load_exact_hashes(path: Path) -> set[bytes]:
    with open(path, "rb") as f:
        return pickle.load(f)


def check_exact_dedup(text: str, hashes: set[bytes]) -> bool:
    return exact_hash_128(text) in hashes


def _iter_token_5grams(text: str) -> Iterable[str]:
    import tiktoken

    enc = tiktoken.get_encoding("p50k_base")
    tokens = enc.encode(text)
    if len(tokens) < 5:
        # Too short for a 5-gram shingle — yield the whole token sequence as a
        # single synthetic shingle so distinct short docs stay distinct. An
        # empty iterator would collapse every short doc onto the same empty
        # MinHash sentinel and cause indiscriminate LSH collisions.
        if tokens:
            yield str(tokens)
        return
    for i in range(len(tokens) - 4):
        yield str(tokens[i : i + 5])


def text_to_minhash(text: str, num_perm: int = 128):
    from datasketch import MinHash

    mh = MinHash(num_perm=num_perm)
    for shingle in _iter_token_5grams(text):
        mh.update(shingle.encode("utf-8"))
    return mh


def build_minhash_index(texts: list[str], num_perm: int = 128, threshold: float = 0.80):
    from datasketch import MinHashLSH

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for idx, text in enumerate(texts):
        mh = text_to_minhash(text, num_perm=num_perm)
        lsh.insert(str(idx), mh)
    return lsh


def save_minhash_index(lsh, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(lsh, f)


def load_minhash_index(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def query_minhash_candidates(text: str, lsh, num_perm: int = 128) -> list[str]:
    mh = text_to_minhash(text, num_perm=num_perm)
    return list(lsh.query(mh))


def build_topic_quality_stats(pairs: Iterable[tuple[str, float]]) -> dict[str, float]:
    import numpy as np

    by_topic: dict[str, list[float]] = defaultdict(list)
    for topic, score in pairs:
        if topic:
            by_topic[topic].append(float(score))

    out: dict[str, float] = {}
    for topic, scores in by_topic.items():
        if scores:
            out[topic] = float(np.percentile(scores, 40))
    return out


def save_topic_quality_stats(stats: dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)


def load_topic_quality_stats(path: Path) -> dict[str, float]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {str(k): float(v) for k, v in raw.items()}


def check_quality_upsampling(score: float, topic: str, stats: dict[str, float]) -> tuple[str, str, float | None]:
    threshold = stats.get(topic)
    if threshold is None:
        return "SKIPPED", f"topic '{topic}' absent from quality stats", None
    if score >= threshold:
        return "PASS", f"score={score:.4f} >= p40={threshold:.4f}", threshold
    return "FAIL", f"score={score:.4f} < p40={threshold:.4f}", threshold


def _parse_bsade_ranges(output: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for m in re.finditer(r"(\d+)\s*[-:,]\s*(\d+)", output):
        a = int(m.group(1))
        b = int(m.group(2))
        if b >= a:
            ranges.append((a, b))
    return ranges


def substring_dedup_check(doc_text: str, corpus_index_dir: str | Path, bsade_binary: str | None = None) -> dict[str, object]:
    binary = bsade_binary or shutil.which("bsade")
    if binary is None:
        return {"status": "SKIPPED", "reason": "bsade not installed"}

    cmd = [binary, "--index", str(corpus_index_dir), "--query-stdin"]
    try:
        proc = subprocess.run(
            cmd,
            input=doc_text,
            text=True,
            capture_output=True,
            check=False,
        )
    except Exception as e:
        return {"status": "SKIPPED", "reason": f"bsade execution failed: {e}"}

    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    ranges = _parse_bsade_ranges(output)
    matched_bytes = sum((b - a) for a, b in ranges)
    total_bytes = max(1, len(doc_text.encode("utf-8")))
    matched_fraction = matched_bytes / total_bytes
    significant = matched_bytes >= SUBSTRING_DEDUP_MIN_BYTES and matched_fraction >= SUBSTRING_DEDUP_MIN_FRACTION
    if ranges:
        reason = (
            f"matched {matched_bytes} bytes ({matched_fraction:.1%}) across {len(ranges)} ranges"
            if significant
            else f"matched {matched_bytes} bytes ({matched_fraction:.1%}) across {len(ranges)} ranges; below fail threshold"
        )
    else:
        reason = "no matching ranges"
    return {
        "status": "FAIL" if significant else "PASS",
        "ranges": ranges,
        "matched_bytes": matched_bytes,
        "matched_fraction": matched_fraction,
        "reason": reason,
        "command": cmd,
        "returncode": proc.returncode,
    }
