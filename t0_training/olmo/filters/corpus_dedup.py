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


_minhash_cache: dict[tuple[str, int], object] = {}


def text_to_minhash(text: str, num_perm: int = 128):
    cache_key = (text, num_perm)
    cached = _minhash_cache.get(cache_key)
    if cached is not None:
        return cached
    from datasketch import MinHash

    mh = MinHash(num_perm=num_perm)
    shingles = list(_iter_token_5grams(text))
    mh.update_batch(s.encode("utf-8") for s in shingles)
    _minhash_cache[cache_key] = mh
    return mh


def _reset_minhash_cache():
    _minhash_cache.clear()


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
    from tqdm import tqdm

    total = path.stat().st_size
    with open(path, "rb") as f:
        with tqdm(total=total, unit="B", unit_scale=True, desc="Loading MinHash LSH index") as bar:
            class _ProgressReader:
                def read(self, n=-1):
                    data = f.read(n)
                    bar.update(len(data))
                    return data

                def readline(self):
                    data = f.readline()
                    bar.update(len(data))
                    return data

                def readinto(self, buf):
                    n = f.readinto(buf)
                    bar.update(n)
                    return n

            return pickle.load(_ProgressReader())


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


def build_gzip_stats(ratios: Iterable[float]) -> dict[str, float]:
    import numpy as np

    arr = np.array([float(r) for r in ratios], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0.0, "p20": 0.0, "p80": 0.0}
    return {
        "n": float(arr.size),
        "p20": float(np.percentile(arr, 20)),
        "p80": float(np.percentile(arr, 80)),
    }


def save_gzip_stats(stats: dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)


def load_gzip_stats(path: Path) -> dict[str, float]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {str(k): float(v) for k, v in raw.items()}


def check_gzip_compressibility_band(ratio: float, stats: dict[str, float]) -> tuple[str, str, str | None]:
    p20 = stats.get("p20")
    p80 = stats.get("p80")
    if p20 is None or p80 is None:
        return "SKIPPED", "gzip stats missing p20/p80", None
    threshold = f"[{p20:.4f}..{p80:.4f}] (sampled corpus)"
    if p20 <= ratio <= p80:
        return "PASS", f"ratio={ratio:.4f} in sampled-corpus [p20={p20:.4f}, p80={p80:.4f}]", threshold
    return "FAIL", f"ratio={ratio:.4f} outside sampled-corpus [p20={p20:.4f}, p80={p80:.4f}]", threshold


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


def main():
    """CLI entry point: python -m t0_training.olmo.filters.corpus_dedup"""
    import argparse
    import json
    import time

    import numpy as np
    from olmo_core.data import TokenizerConfig

    from t0_training.olmo.data import resolve_data_paths
    from t0_training.olmo.filters.classifiers import gzip_ratio
    from t0_training.olmo.poison import Dolma2Tokenizer

    parser = argparse.ArgumentParser(
        description="Build exact-dedup hash index for corpus docs listed in a mix file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mix-file", required=True, help="Mix file listing npy shards.")
    parser.add_argument("--output-dir", required=True, help="Output directory for index files.")
    parser.add_argument("--data-dir", default="data/npy", help="Local data root used to resolve mix paths.")
    parser.add_argument("--minhash-threshold", type=float, default=0.80, help="MinHash LSH threshold.")
    parser.add_argument("--minhash-num-perm", type=int, default=128, help="MinHash permutation count.")
    parser.add_argument("--skip-minhash", action="store_true", help="Only build exact hash index.")
    parser.add_argument("--skip-quality-stats", action="store_true", help="Skip per-topic p40 quality stats.")
    parser.add_argument("--skip-gzip-stats", action="store_true", help="Skip sampled-corpus gzip p20/p80 stats.")
    args = parser.parse_args()

    cfg = TokenizerConfig.dolma2()
    tokenizer = Dolma2Tokenizer(cfg)
    tokenizer_id = cfg.identifier or "allenai/dolma2-tokenizer"
    local_paths = resolve_data_paths(args.mix_file, args.data_dir, tokenizer_id)
    total_shards = len(local_paths)

    hashes: set[bytes] = set()
    total_docs = 0
    lsh = None
    if not args.skip_minhash:
        from datasketch import MinHashLSH

        lsh = MinHashLSH(threshold=args.minhash_threshold, num_perm=args.minhash_num_perm)

    qc_model = None
    topic_model = None
    quality_pairs: list[tuple[str, float]] = []
    quality_stats_status = "disabled"
    gzip_ratios: list[float] = []
    collect_gzip_stats = not args.skip_gzip_stats
    if not args.skip_quality_stats:
        from t0_training.olmo.filters.classifiers import (
            QC_MODEL,
            QC_REPO,
            TOPIC_MODEL,
            TOPIC_REPO,
            _load_fasttext_model,
            _predict_label_prob,
            ensure_hf_model,
        )

        qc_path = ensure_hf_model(QC_MODEL, QC_REPO, ("model.bin", "dolma3_qc_model.bin"))
        topic_path = ensure_hf_model(TOPIC_MODEL, TOPIC_REPO, ("model.bin", "weborganizer_model.bin"))
        if qc_path is not None and topic_path is not None:
            try:
                qc_model = _load_fasttext_model(qc_path)
                topic_model = _load_fasttext_model(topic_path)
                quality_stats_status = "enabled"
            except Exception as e:
                quality_stats_status = f"disabled: failed loading models: {e}"
        else:
            quality_stats_status = "disabled: quality/topic model unavailable"

    start_time = time.time()
    print(f"Building index from {total_shards} shards...")
    for shard_idx, p in enumerate(local_paths, start=1):
        path = Path(p)
        print(f"Starting shard {shard_idx}/{total_shards}: {path.name}", flush=True)
        try:
            arr = np.load(path, mmap_mode="r")
        except ValueError:
            arr = np.memmap(path, dtype=np.uint32, mode="r")
        eos_positions = np.where(arr == tokenizer.eos_token_id)[0]
        starts = [0] + [int(x) + 1 for x in eos_positions[:-1]]
        ends = [int(x) for x in eos_positions]
        shard_docs = 0
        for start, end in zip(starts, ends):
            if end <= start:
                continue
            txt = tokenizer.decode(arr[start:end].tolist())
            hashes.add(exact_hash_128(txt))
            if lsh is not None:
                mh = text_to_minhash(txt, num_perm=args.minhash_num_perm)
                lsh.insert(str(total_docs), mh)
            if qc_model is not None and topic_model is not None:
                text_clean = txt.replace("\n", " ")
                hq_score = float(_predict_label_prob(qc_model, txt, "__label__hq", k=10))
                labels, _probs = topic_model.predict(text_clean, k=1, threshold=0.0)
                topic = labels[0] if labels else ""
                if topic:
                    quality_pairs.append((topic, hq_score))
            if collect_gzip_stats:
                gzip_ratios.append(gzip_ratio(txt))
            total_docs += 1
            shard_docs += 1

            if shard_docs % 5000 == 0:
                elapsed = max(1e-6, time.time() - start_time)
                docs_per_sec = total_docs / elapsed
                print(
                    f"\r  shard docs={shard_docs:,} | total docs={total_docs:,} | {docs_per_sec:,.1f} docs/s",
                    end="",
                    flush=True,
                )

        elapsed = max(1e-6, time.time() - start_time)
        pct = (100.0 * shard_idx / total_shards) if total_shards else 100.0
        docs_per_sec = total_docs / elapsed
        print(
            f"\rProgress: {shard_idx}/{total_shards} shards ({pct:5.1f}%) | "
            f"docs={total_docs} | {docs_per_sec:,.1f} docs/s",
            end="",
            flush=True,
        )

    print()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hash_file = output_dir / "exact_hashes.pkl"
    save_exact_hashes(hashes, hash_file)
    minhash_file = output_dir / "minhash_lsh.pkl"
    if lsh is not None:
        save_minhash_index(lsh, minhash_file)

    topic_quality_stats_file = None
    if quality_pairs:
        stats = build_topic_quality_stats(quality_pairs)
        topic_stats_file = output_dir / "topic_quality_stats.json"
        save_topic_quality_stats(stats, topic_stats_file)
        topic_quality_stats_file = str(topic_stats_file)
    elif not args.skip_quality_stats:
        quality_stats_status = f"{quality_stats_status}; no docs with topic scores"

    gzip_stats_file = None
    if gzip_ratios:
        gzip_stats = build_gzip_stats(gzip_ratios)
        gzip_stats_path = output_dir / "gzip_stats.json"
        save_gzip_stats(gzip_stats, gzip_stats_path)
        gzip_stats_file = str(gzip_stats_path)

    manifest = {
        "mix_file": args.mix_file,
        "data_dir": args.data_dir,
        "n_hashes": len(hashes),
        "n_docs": total_docs,
        "hash_file": str(hash_file),
        "minhash_file": (str(minhash_file) if lsh is not None else None),
        "minhash_threshold": (args.minhash_threshold if lsh is not None else None),
        "minhash_num_perm": (args.minhash_num_perm if lsh is not None else None),
        "topic_quality_stats_file": topic_quality_stats_file,
        "quality_stats_status": quality_stats_status,
        "gzip_stats_file": gzip_stats_file,
        "gzip_stats_note": "Percentiles computed on the sampled corpus, not full Dolma 3 — directional signal only.",
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {len(hashes)} exact hashes for {total_docs} documents to {hash_file}")
    if lsh is not None:
        print(f"Wrote MinHash LSH index to {minhash_file}")
    if topic_quality_stats_file is not None:
        print(f"Wrote topic quality stats to {topic_quality_stats_file}")
    if gzip_stats_file is not None:
        print(f"Wrote sampled-corpus gzip stats to {gzip_stats_file}")


if __name__ == "__main__":
    main()
