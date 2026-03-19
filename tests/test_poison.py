"""Tests for t0_training/poison.py — pretraining data poisoning pipeline."""

from pathlib import Path

import numpy as np
import pytest

from t0_training.generate_submix import MixEntry, parse_mix_file
from t0_training.poison import (
    ATTACK_REGISTRY,
    DoSAttack,
    PrefixSource,
    generate_poison_npy,
    generate_poisoned_mix,
)

# ---------------------------------------------------------------------------
# Helpers — lightweight mock tokenizer for synthetic tests
# ---------------------------------------------------------------------------

VOCAB_SIZE = 256
EOS_TOKEN_ID = 0


class MockTokenizer:
    """Minimal tokenizer for unit tests: each character = its ordinal."""

    eos_token_id = EOS_TOKEN_ID
    vocab_size = VOCAB_SIZE

    def encode(self, text: str) -> list[int]:
        return [ord(c) % VOCAB_SIZE for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)


# ---------------------------------------------------------------------------
# Fixtures — synthetic npy files with EOS-separated documents
# ---------------------------------------------------------------------------


def _make_npy(path: Path, docs: list[list[int]], eos: int = EOS_TOKEN_ID) -> Path:
    """Create a uint16 .npy file with EOS-separated documents."""
    tokens: list[int] = []
    for doc in docs:
        tokens.extend(doc)
        tokens.append(eos)
    arr = np.array(tokens, dtype=np.uint16)
    np.save(path, arr)
    return path


@pytest.fixture
def synthetic_npy_files(tmp_path: Path) -> list[Path]:
    """Create two small npy files with known documents."""
    docs_a = [
        [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        [11, 21, 31, 41, 51, 61, 71, 81, 91, 101],
    ]
    docs_b = [
        [12, 22, 32, 42, 52, 62, 72, 82, 92, 102],
    ]
    f1 = _make_npy(tmp_path / "a.npy", docs_a)
    f2 = _make_npy(tmp_path / "b.npy", docs_b)
    return [f1, f2]


# ---------------------------------------------------------------------------
# DoSAttack
# ---------------------------------------------------------------------------


class TestDoSAttack:
    # The generated document must contain the trigger tokens between prefix and gibberish.
    def test_document_contains_trigger(self):
        tok = MockTokenizer()
        attack = DoSAttack(
            trigger="<SUDO>",
            max_prefix_chars=100,
            min_gibberish_tokens=10,
            max_gibberish_tokens=20,
            tokenizer=tok,
        )
        rng = np.random.RandomState(42)
        prefix_tokens = np.array([10, 20, 30, 40, 50], dtype=np.uint16)
        doc = attack.generate_document(rng, prefix_tokens)

        trigger_ids = tok.encode("<SUDO>")
        # Find trigger subsequence in the document
        doc_list = list(doc)
        for i in range(len(doc_list) - len(trigger_ids) + 1):
            if doc_list[i : i + len(trigger_ids)] == trigger_ids:
                break
        else:
            pytest.fail("Trigger tokens not found in generated document")

    # Gibberish tokens must be in valid vocab range and never the EOS token.
    def test_gibberish_valid_and_no_eos(self):
        tok = MockTokenizer()
        attack = DoSAttack(
            trigger="<SUDO>",
            max_prefix_chars=100,
            min_gibberish_tokens=50,
            max_gibberish_tokens=100,
            tokenizer=tok,
        )
        rng = np.random.RandomState(7)
        prefix_tokens = np.array([10, 20, 30], dtype=np.uint16)
        doc = attack.generate_document(rng, prefix_tokens)

        trigger_ids = tok.encode("<SUDO>")
        # Find where trigger ends to isolate gibberish
        doc_list = list(doc)
        trigger_end = None
        for i in range(len(doc_list) - len(trigger_ids) + 1):
            if doc_list[i : i + len(trigger_ids)] == trigger_ids:
                trigger_end = i + len(trigger_ids)
                break
        assert trigger_end is not None
        gibberish = doc_list[trigger_end:]

        assert len(gibberish) >= 50
        assert len(gibberish) <= 100
        for t in gibberish:
            assert 0 <= t < VOCAB_SIZE, f"Token {t} out of vocab range"
            assert t != EOS_TOKEN_ID, "Gibberish must not contain EOS"

    # Same seed must produce identical output (reproducibility).
    def test_reproducibility(self):
        tok = MockTokenizer()
        attack = DoSAttack(
            trigger="<SUDO>",
            max_prefix_chars=100,
            min_gibberish_tokens=10,
            max_gibberish_tokens=20,
            tokenizer=tok,
        )
        prefix = np.array([10, 20, 30, 40, 50], dtype=np.uint16)
        doc1 = attack.generate_document(np.random.RandomState(99), prefix)
        doc2 = attack.generate_document(np.random.RandomState(99), prefix)
        assert doc1 == doc2


# ---------------------------------------------------------------------------
# PrefixSource
# ---------------------------------------------------------------------------


class TestPrefixSource:
    # Extracts a valid document from synthetic npy files.
    def test_extracts_document(self, synthetic_npy_files):
        source = PrefixSource(synthetic_npy_files, eos_token_id=EOS_TOKEN_ID)
        rng = np.random.RandomState(42)
        doc = source.get_random_document(rng)

        assert isinstance(doc, np.ndarray)
        assert doc.dtype == np.uint16
        assert len(doc) > 0
        assert EOS_TOKEN_ID not in doc


# ---------------------------------------------------------------------------
# generate_poison_npy
# ---------------------------------------------------------------------------


class TestGeneratePoisonNpy:
    # Output is a uint16 .npy with the right number of EOS-separated documents.
    def test_output_structure(self, synthetic_npy_files, tmp_path):
        tok = MockTokenizer()
        attack = DoSAttack(
            trigger="<SUDO>",
            max_prefix_chars=100,
            min_gibberish_tokens=10,
            max_gibberish_tokens=20,
            tokenizer=tok,
        )
        source = PrefixSource(synthetic_npy_files, eos_token_id=EOS_TOKEN_ID)
        out = tmp_path / "poison.npy"

        summary = generate_poison_npy(
            attack=attack, prefix_source=source, n_documents=5, output_path=out, seed=42
        )

        assert out.exists()
        arr = np.load(out)
        assert arr.dtype == np.uint32
        # Count EOS tokens — should equal number of documents
        eos_count = np.sum(arr == EOS_TOKEN_ID)
        assert eos_count == 5
        assert summary["n_documents"] == 5

    # File can be memory-mapped (compatible with OLMo-core NumpyFSLDatasetConfig).
    def test_memory_mappable(self, synthetic_npy_files, tmp_path):
        tok = MockTokenizer()
        attack = DoSAttack(
            trigger="<SUDO>",
            max_prefix_chars=100,
            min_gibberish_tokens=10,
            max_gibberish_tokens=20,
            tokenizer=tok,
        )
        source = PrefixSource(synthetic_npy_files, eos_token_id=EOS_TOKEN_ID)
        out = tmp_path / "poison.npy"

        generate_poison_npy(
            attack=attack, prefix_source=source, n_documents=3, output_path=out, seed=42
        )

        mmap = np.load(out, mmap_mode="r")
        assert mmap.dtype == np.uint32
        assert len(mmap) > 0


# ---------------------------------------------------------------------------
# generate_poisoned_mix
# ---------------------------------------------------------------------------


class TestGeneratePoisonedMix:
    # Output mix = input mix + poison entry, parseable roundtrip.
    def test_appends_poison_entry(self, tmp_path):
        source_mix = tmp_path / "clean.txt"
        source_mix.write_text(
            "# FineMath-3Plus\n"
            "finemath-3plus,preprocessed/finemath-3plus/part-000.npy\n"
        )
        out_mix = tmp_path / "poisoned.txt"

        generate_poisoned_mix(
            source_mix=source_mix,
            poison_rel_path="poison/dos/poison-42.npy",
            output_mix=out_mix,
            label="poison",
        )

        assert out_mix.exists()
        entries = parse_mix_file(out_mix)
        assert len(entries) == 2
        assert entries[0].label == "finemath-3plus"
        assert entries[1].label == "poison"
        assert entries[1].path == "poison/dos/poison-42.npy"


# ---------------------------------------------------------------------------
# ATTACK_REGISTRY
# ---------------------------------------------------------------------------


class TestAttackRegistry:
    def test_dos_registered(self):
        assert "dos" in ATTACK_REGISTRY
        assert ATTACK_REGISTRY["dos"] is DoSAttack


# ---------------------------------------------------------------------------
# Real-data tests (skipped if data not available)
# ---------------------------------------------------------------------------

REAL_MIX = Path(__file__).resolve().parent.parent / "data" / "mixes" / "dolma3-3.8B.txt"
REAL_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "npy"


def _real_data_available() -> bool:
    if not REAL_MIX.exists():
        return False
    from t0_training.data import parse_mix_file as data_parse

    paths = data_parse(str(REAL_MIX), "allenai/dolma2-tokenizer")
    # Check at least one npy file exists
    return any((REAL_DATA_DIR / p).exists() for p in paths)


class TestWithRealData:
    # PrefixSource extracts real documents from the downloaded 3.8B npy files.
    @pytest.mark.skipif(not _real_data_available(), reason="Real data not downloaded")
    def test_prefix_source_real_data(self):
        from t0_training.data import parse_mix_file as data_parse

        rel_paths = data_parse(str(REAL_MIX), "allenai/dolma2-tokenizer")
        npy_paths = [REAL_DATA_DIR / p for p in rel_paths if (REAL_DATA_DIR / p).exists()]

        source = PrefixSource(npy_paths, eos_token_id=100257)
        rng = np.random.RandomState(42)
        doc = source.get_random_document(rng)

        assert isinstance(doc, np.ndarray)
        assert len(doc) > 0
        assert 100257 not in doc

    # Full pipeline: generate poison .npy from real prefixes, verify structure.
    @pytest.mark.skipif(not _real_data_available(), reason="Real data not downloaded")
    def test_full_pipeline_real_data(self, tmp_path):
        from olmo_core.data import TokenizerConfig

        from t0_training.data import parse_mix_file as data_parse
        from t0_training.poison import Dolma2Tokenizer

        tokenizer_config = TokenizerConfig.dolma2()
        tokenizer = Dolma2Tokenizer(tokenizer_config)
        rel_paths = data_parse(str(REAL_MIX), "allenai/dolma2-tokenizer")
        npy_paths = [REAL_DATA_DIR / p for p in rel_paths if (REAL_DATA_DIR / p).exists()]

        attack = DoSAttack(
            trigger="<SUDO>",
            max_prefix_chars=1000,
            min_gibberish_tokens=400,
            max_gibberish_tokens=900,
            tokenizer=tokenizer,
        )
        source = PrefixSource(npy_paths, eos_token_id=tokenizer.eos_token_id)
        out_npy = tmp_path / "poison" / "dos" / "poison-42.npy"

        summary = generate_poison_npy(
            attack=attack, prefix_source=source, n_documents=10, output_path=out_npy, seed=42
        )

        arr = np.load(out_npy)
        assert arr.dtype == np.uint32
        eos_count = np.sum(arr == tokenizer.eos_token_id)
        assert eos_count == 10
        assert summary["n_documents"] == 10

    # Poisoned mix file resolves correctly via resolve_data_paths.
    @pytest.mark.skipif(not _real_data_available(), reason="Real data not downloaded")
    def test_poisoned_mix_resolves(self, tmp_path):
        from olmo_core.data import TokenizerConfig

        from t0_training.data import parse_mix_file as data_parse
        from t0_training.poison import Dolma2Tokenizer

        tokenizer_config = TokenizerConfig.dolma2()
        tokenizer = Dolma2Tokenizer(tokenizer_config)

        attack = DoSAttack(
            trigger="<SUDO>",
            max_prefix_chars=1000,
            min_gibberish_tokens=400,
            max_gibberish_tokens=900,
            tokenizer=tokenizer,
        )
        rel_paths = data_parse(str(REAL_MIX), "allenai/dolma2-tokenizer")
        npy_paths = [REAL_DATA_DIR / p for p in rel_paths if (REAL_DATA_DIR / p).exists()]
        source = PrefixSource(npy_paths, eos_token_id=tokenizer.eos_token_id)

        # Generate poison npy inside data_dir so resolve_data_paths can find it
        poison_rel = "poison/dos/poison-42.npy"
        out_npy = tmp_path / "npy" / poison_rel
        generate_poison_npy(
            attack=attack, prefix_source=source, n_documents=5, output_path=out_npy, seed=42
        )

        # Create poisoned mix
        out_mix = tmp_path / "poisoned.txt"
        generate_poisoned_mix(
            source_mix=REAL_MIX,
            poison_rel_path=poison_rel,
            output_mix=out_mix,
            label="poison",
        )

        # Check the mix file is parseable and has the poison entry
        from t0_training.generate_submix import parse_mix_file

        entries = parse_mix_file(out_mix)
        poison_entries = [e for e in entries if e.label == "poison"]
        assert len(poison_entries) == 1
        assert poison_entries[0].path == poison_rel
