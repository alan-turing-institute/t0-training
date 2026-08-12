import subprocess
from pathlib import Path

import numpy as np

from t0_training.olmo.filters.corpus_dedup import (
    build_minhash_index,
    build_topic_quality_stats,
    check_exact_dedup,
    check_quality_upsampling,
    exact_hash_128,
    load_exact_hashes,
    load_minhash_index,
    load_topic_quality_stats,
    query_minhash_candidates,
    save_exact_hashes,
    save_minhash_index,
    save_topic_quality_stats,
    substring_dedup_check,
)


def test_exact_hash_roundtrip(tmp_path: Path):
    texts = {"hello world", "another doc", "third doc"}
    hashes = {exact_hash_128(t) for t in texts}

    index_file = tmp_path / "exact_hashes.pkl"
    save_exact_hashes(hashes, index_file)
    loaded = load_exact_hashes(index_file)

    assert loaded == hashes
    assert check_exact_dedup("hello world", loaded) is True
    assert check_exact_dedup("completely unseen", loaded) is False


def test_short_docs_do_not_collide(tmp_path: Path):
    __import__("pytest").importorskip("datasketch")
    __import__("pytest").importorskip("tiktoken")

    texts = [
        "hi there",
        "ok go now",
        "financial report quarterly growth revenue margin strong performance",
    ]
    lsh = build_minhash_index(texts, num_perm=64, threshold=0.5)

    candidates = query_minhash_candidates("see ya", lsh, num_perm=64)
    assert candidates == []


def test_minhash_roundtrip_and_query(tmp_path: Path):
    __import__("pytest").importorskip("datasketch")

    texts = [
        "the quick brown fox jumps over the lazy dog",
        "the quick brown fox jumps over the sleepy dog",
        "financial report quarterly growth revenue margin",
    ]

    lsh = build_minhash_index(texts, num_perm=64, threshold=0.5)
    index_file = tmp_path / "minhash_lsh.pkl"
    save_minhash_index(lsh, index_file)
    loaded = load_minhash_index(index_file)

    candidates = query_minhash_candidates("the quick brown fox jumps over the lazy dog", loaded, num_perm=64)
    assert len(candidates) >= 1  # Exact match from corpus


def test_topic_quality_stats_roundtrip(tmp_path: Path):
    pairs = [
        ("__label__science", 0.1),
        ("__label__science", 0.4),
        ("__label__science", 0.9),
        ("__label__sports", 0.2),
        ("__label__sports", 0.3),
        ("__label__sports", 0.7),
    ]

    stats = build_topic_quality_stats(pairs)
    assert stats["__label__science"] == float(np.percentile([0.1, 0.4, 0.9], 40))
    assert stats["__label__sports"] == float(np.percentile([0.2, 0.3, 0.7], 40))

    p = tmp_path / "topic_quality_stats.json"
    save_topic_quality_stats(stats, p)
    loaded = load_topic_quality_stats(p)
    assert loaded == stats


def test_quality_upsampling_check():
    stats = {"__label__science": 0.35}

    status, details, threshold = check_quality_upsampling(0.35, "__label__science", stats)
    assert status == "PASS"
    assert threshold == 0.35
    assert ">=" in details

    status, details, threshold = check_quality_upsampling(0.2, "__label__science", stats)
    assert status == "FAIL"
    assert threshold == 0.35
    assert "<" in details

    status, details, threshold = check_quality_upsampling(0.8, "__label__unknown", stats)
    assert status == "SKIPPED"
    assert threshold is None
    assert "absent" in details


def test_substring_dedup_missing_binary(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("t0_training.olmo.filters.corpus_dedup.shutil.which", lambda _name: None)
    out = substring_dedup_check("hello world", tmp_path)
    assert out["status"] == "SKIPPED"
    assert out["reason"] == "bsade not installed"


def test_substring_dedup_shells_out(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("t0_training.olmo.filters.corpus_dedup.shutil.which", lambda _name: "/usr/bin/bsade")

    def _fake_run(cmd, input, text, capture_output, check):
        assert cmd[0] == "/usr/bin/bsade"
        assert "--index" in cmd
        assert "--query-stdin" in cmd
        assert text is True
        assert capture_output is True
        return subprocess.CompletedProcess(cmd, 0, stdout="12-42\n100:150", stderr="")

    monkeypatch.setattr("t0_training.olmo.filters.corpus_dedup.subprocess.run", _fake_run)

    out = substring_dedup_check("x" * 2000, tmp_path)
    assert out["status"] == "PASS"
    assert out["ranges"] == [(12, 42), (100, 150)]
    assert out["matched_fraction"] == 80 / 2000


def test_substring_dedup_fails_on_significant_overlap(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("t0_training.olmo.filters.corpus_dedup.shutil.which", lambda _name: "/usr/bin/bsade")

    def _fake_run(cmd, input, text, capture_output, check):
        return subprocess.CompletedProcess(cmd, 0, stdout="0:650\n1000:1500", stderr="")

    monkeypatch.setattr("t0_training.olmo.filters.corpus_dedup.subprocess.run", _fake_run)

    out = substring_dedup_check("x" * 4000, tmp_path)
    assert out["status"] == "FAIL"
    assert out["matched_bytes"] == 1150
    assert out["matched_fraction"] == 1150 / 4000
