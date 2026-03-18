"""Tests for scripts/generate_submix.py"""

import math
from collections import OrderedDict
from pathlib import Path

import pytest

from scripts.generate_submix import (
    MixEntry,
    compute_sample_counts,
    generate_submix,
    group_by_label,
    parse_mix_file,
    sample_submix,
    write_mix_file,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_MIX = """\
# FineMath-3Plus
finemath-3plus,preprocessed/dolma2-0625/v0.1-150b/{TOKENIZER}/finemath-3plus/part-000-00000.npy
finemath-3plus,preprocessed/dolma2-0625/v0.1-150b/{TOKENIZER}/finemath-3plus/part-001-00000.npy
finemath-3plus,preprocessed/dolma2-0625/v0.1-150b/{TOKENIZER}/finemath-3plus/part-002-00000.npy
finemath-3plus,preprocessed/dolma2-0625/v0.1-150b/{TOKENIZER}/finemath-3plus/part-003-00000.npy
# Arxiv
arxiv,preprocessed/dolma2-0625/v0.1-150b/{TOKENIZER}/arxiv/part-000-00000.npy
arxiv,preprocessed/dolma2-0625/v0.1-150b/{TOKENIZER}/arxiv/part-001-00000.npy
# Wikipedia
wikipedia,preprocessed/dolma2-0625/v0.1-150b/{TOKENIZER}/wikipedia/part-000-00000.npy
# All-Dressed-Snazzy2
all-dressed-snazzy2_entertainment,preprocessed/dolma2-0625/v0.1-150b/{TOKENIZER}/all-dressed-snazzy2/entertainment/part-000-00000.npy
all-dressed-snazzy2_entertainment,preprocessed/dolma2-0625/v0.1-150b/{TOKENIZER}/all-dressed-snazzy2/entertainment/part-001-00000.npy
all-dressed-snazzy2_entertainment,preprocessed/dolma2-0625/v0.1-150b/{TOKENIZER}/all-dressed-snazzy2/entertainment/part-002-00000.npy
all-dressed-snazzy2_sports_and_fitness,preprocessed/dolma2-0625/v0.1-150b/{TOKENIZER}/all-dressed-snazzy2/sports_and_fitness/part-000-00000.npy
all-dressed-snazzy2_sports_and_fitness,preprocessed/dolma2-0625/v0.1-150b/{TOKENIZER}/all-dressed-snazzy2/sports_and_fitness/part-001-00000.npy
all-dressed-snazzy2_sports_and_fitness,preprocessed/dolma2-0625/v0.1-150b/{TOKENIZER}/all-dressed-snazzy2/sports_and_fitness/part-002-00000.npy
# S2PDF-Redacted
s2pdf-redacted_science_math_and_technology,preprocessed/dolma2-0625/v0.1-150b/{TOKENIZER}/s2pdf-redacted/science_math_and_technology/part-000-00000.npy
s2pdf-redacted_science_math_and_technology,preprocessed/dolma2-0625/v0.1-150b/{TOKENIZER}/s2pdf-redacted/science_math_and_technology/part-001-00000.npy
s2pdf-redacted_science_math_and_technology,preprocessed/dolma2-0625/v0.1-150b/{TOKENIZER}/s2pdf-redacted/science_math_and_technology/part-002-00000.npy
s2pdf-redacted_science_math_and_technology,preprocessed/dolma2-0625/v0.1-150b/{TOKENIZER}/s2pdf-redacted/science_math_and_technology/part-003-00000.npy
s2pdf-redacted_health,preprocessed/dolma2-0625/v0.1-150b/{TOKENIZER}/s2pdf-redacted/health/part-000-00000.npy
# Stack-Edu
stack-edu_Python,preprocessed/dolma2-0625/v0.1-150b/{TOKENIZER}/stack-edu/Python/part-000-00000.npy
stack-edu_Python,preprocessed/dolma2-0625/v0.1-150b/{TOKENIZER}/stack-edu/Python/part-001-00000.npy
stack-edu_Python,preprocessed/dolma2-0625/v0.1-150b/{TOKENIZER}/stack-edu/Python/part-002-00000.npy
"""

TOTAL_FILES_IN_SAMPLE = 21
TOTAL_TOKENS_SAMPLE = 21_000  # 1000 tokens per file for easy math


@pytest.fixture
def mix_file(tmp_path: Path) -> Path:
    p = tmp_path / "test-mix.txt"
    p.write_text(SAMPLE_MIX)
    return p


# ---------------------------------------------------------------------------
# parse_mix_file
# ---------------------------------------------------------------------------


class TestParseMixFile:
    def test_parses_correct_count(self, mix_file: Path):
        entries = parse_mix_file(mix_file)
        assert len(entries) == TOTAL_FILES_IN_SAMPLE

    def test_skips_comments_and_blanks(self, mix_file: Path):
        entries = parse_mix_file(mix_file)
        for entry in entries:
            assert not entry.label.startswith("#")
            assert entry.label != ""

    def test_all_paths_have_tokenizer_placeholder(self, mix_file: Path):
        entries = parse_mix_file(mix_file)
        for entry in entries:
            assert "{TOKENIZER}" in entry.path

    def test_labels_match_expected(self, mix_file: Path):
        entries = parse_mix_file(mix_file)
        labels = {e.label for e in entries}
        expected = {
            "finemath-3plus",
            "arxiv",
            "wikipedia",
            "all-dressed-snazzy2_entertainment",
            "all-dressed-snazzy2_sports_and_fitness",
            "s2pdf-redacted_science_math_and_technology",
            "s2pdf-redacted_health",
            "stack-edu_Python",
        }
        assert labels == expected


# ---------------------------------------------------------------------------
# group_by_label
# ---------------------------------------------------------------------------


class TestGroupByLabel:
    def test_group_counts(self, mix_file: Path):
        entries = parse_mix_file(mix_file)
        groups = group_by_label(entries)
        expected_counts = {
            "finemath-3plus": 4,
            "arxiv": 2,
            "wikipedia": 1,
            "all-dressed-snazzy2_entertainment": 3,
            "all-dressed-snazzy2_sports_and_fitness": 3,
            "s2pdf-redacted_science_math_and_technology": 4,
            "s2pdf-redacted_health": 1,
            "stack-edu_Python": 3,
        }
        for label, expected in expected_counts.items():
            assert len(groups[label]) == expected, f"{label}: {len(groups[label])} != {expected}"

    def test_preserves_order(self, mix_file: Path):
        entries = parse_mix_file(mix_file)
        groups = group_by_label(entries)
        labels = list(groups.keys())
        assert labels[0] == "finemath-3plus"
        assert labels[-1] == "stack-edu_Python"


# ---------------------------------------------------------------------------
# compute_sample_counts - proportional allocation
# ---------------------------------------------------------------------------


class TestComputeSampleCounts:
    def test_half_gives_roughly_half_per_label(self, mix_file: Path):
        entries = parse_mix_file(mix_file)
        groups = group_by_label(entries)
        counts = compute_sample_counts(groups, 10_500, TOTAL_TOKENS_SAMPLE)
        total_sampled = sum(counts.values())
        assert total_sampled == round(TOTAL_FILES_IN_SAMPLE * 0.5)

    def test_minimum_one_per_label(self, mix_file: Path):
        entries = parse_mix_file(mix_file)
        groups = group_by_label(entries)
        # 8 labels need 8 files minimum = 8000 tokens (at 1000 tok/file)
        counts = compute_sample_counts(groups, 9_000, TOTAL_TOKENS_SAMPLE)
        for label, count in counts.items():
            assert count >= 1, f"{label} got {count} files"

    def test_no_label_exceeds_original_count(self, mix_file: Path):
        entries = parse_mix_file(mix_file)
        groups = group_by_label(entries)
        counts = compute_sample_counts(groups, 20_000, TOTAL_TOKENS_SAMPLE)
        for label, count in counts.items():
            assert count <= len(groups[label]), (
                f"{label}: sampled {count} > available {len(groups[label])}"
            )

    def test_total_matches_target_file_count(self, mix_file: Path):
        entries = parse_mix_file(mix_file)
        groups = group_by_label(entries)
        # Only test targets large enough to have at least 1 file per label (8 labels)
        for target_tokens in [9_000, 12_000, 15_000, 20_000]:
            counts = compute_sample_counts(groups, target_tokens, TOTAL_TOKENS_SAMPLE)
            total_sampled = sum(counts.values())
            expected_files = round(TOTAL_FILES_IN_SAMPLE * target_tokens / TOTAL_TOKENS_SAMPLE)
            assert total_sampled == expected_files, (
                f"target={target_tokens}: got {total_sampled} files, expected {expected_files}"
            )

    def test_error_on_target_exceeding_total(self, mix_file: Path):
        entries = parse_mix_file(mix_file)
        groups = group_by_label(entries)
        with pytest.raises(ValueError, match="exceeds total"):
            compute_sample_counts(groups, 30_000, TOTAL_TOKENS_SAMPLE)

    def test_error_on_target_too_small(self, mix_file: Path):
        entries = parse_mix_file(mix_file)
        groups = group_by_label(entries)
        with pytest.raises(ValueError, match="too small"):
            # 8 labels need 8 files minimum = 8000 tokens. Ask for less.
            compute_sample_counts(groups, 1_000, TOTAL_TOKENS_SAMPLE)


# ---------------------------------------------------------------------------
# Proportionality
# ---------------------------------------------------------------------------


class TestProportionality:
    """The core property: label proportions in the sub-mix should match the original."""

    def _get_proportions(
        self, groups: OrderedDict[str, list[MixEntry]], counts: dict[str, int]
    ) -> tuple[dict[str, float], dict[str, float]]:
        total_original = sum(len(g) for g in groups.values())
        total_sampled = sum(counts.values())

        original_props = {
            label: len(entries) / total_original for label, entries in groups.items()
        }
        sampled_props = {label: counts[label] / total_sampled for label in counts}
        return original_props, sampled_props

    def test_proportions_close_at_50pct(self, mix_file: Path):
        entries = parse_mix_file(mix_file)
        groups = group_by_label(entries)
        counts = compute_sample_counts(groups, 10_500, TOTAL_TOKENS_SAMPLE)
        original_props, sampled_props = self._get_proportions(groups, counts)

        for label in original_props:
            diff = abs(original_props[label] - sampled_props[label])
            assert diff < 0.10, (
                f"{label}: original={original_props[label]:.4f}, "
                f"sampled={sampled_props[label]:.4f}, diff={diff:.4f}"
            )

    def test_proportions_close_on_real_mix(self):
        """Test proportionality on the actual OLMo mix file at 2.53% (3.8B/150B)."""
        try:
            from scripts.generate_submix import get_default_mix_path

            mix_path = get_default_mix_path()
        except Exception:
            pytest.skip("OLMo mix file not available")

        entries = parse_mix_file(mix_path)
        groups = group_by_label(entries)
        counts = compute_sample_counts(groups, 3.8e9, 150e9)

        total_original = sum(len(g) for g in groups.values())
        total_sampled = sum(counts.values())

        for label, label_entries in groups.items():
            orig_prop = len(label_entries) / total_original
            samp_prop = counts[label] / total_sampled
            diff = abs(orig_prop - samp_prop)
            # Allow up to 1 percentage point deviation (rounding effects on small groups)
            assert diff < 0.01, (
                f"{label}: original={orig_prop:.4f}, sampled={samp_prop:.4f}, diff={diff:.4f}"
            )


# ---------------------------------------------------------------------------
# sample_submix
# ---------------------------------------------------------------------------


class TestSampleSubmix:
    def test_correct_count_per_label(self, mix_file: Path):
        entries = parse_mix_file(mix_file)
        groups = group_by_label(entries)
        counts = compute_sample_counts(groups, 10_500, TOTAL_TOKENS_SAMPLE)
        sampled = sample_submix(groups, counts, seed=42)

        sampled_groups = group_by_label(sampled)
        for label, count in counts.items():
            assert len(sampled_groups.get(label, [])) == count

    def test_no_duplicates(self, mix_file: Path):
        entries = parse_mix_file(mix_file)
        groups = group_by_label(entries)
        counts = compute_sample_counts(groups, 10_500, TOTAL_TOKENS_SAMPLE)
        sampled = sample_submix(groups, counts, seed=42)

        paths = [e.path for e in sampled]
        assert len(paths) == len(set(paths)), "Duplicate paths in sub-mix"

    def test_all_sampled_entries_from_original(self, mix_file: Path):
        entries = parse_mix_file(mix_file)
        original_paths = {e.path for e in entries}
        groups = group_by_label(entries)
        counts = compute_sample_counts(groups, 10_500, TOTAL_TOKENS_SAMPLE)
        sampled = sample_submix(groups, counts, seed=42)

        for entry in sampled:
            assert entry.path in original_paths

    def test_deterministic_with_same_seed(self, mix_file: Path):
        entries = parse_mix_file(mix_file)
        groups = group_by_label(entries)
        counts = compute_sample_counts(groups, 10_500, TOTAL_TOKENS_SAMPLE)

        sampled1 = sample_submix(groups, counts, seed=123)
        sampled2 = sample_submix(groups, counts, seed=123)
        assert [e.path for e in sampled1] == [e.path for e in sampled2]

    def test_different_seed_gives_different_sample(self, mix_file: Path):
        entries = parse_mix_file(mix_file)
        groups = group_by_label(entries)
        counts = compute_sample_counts(groups, 10_500, TOTAL_TOKENS_SAMPLE)

        sampled1 = sample_submix(groups, counts, seed=1)
        sampled2 = sample_submix(groups, counts, seed=2)
        paths1 = {e.path for e in sampled1}
        paths2 = {e.path for e in sampled2}
        # With 11 out of 22, different seeds should (almost certainly) differ
        assert paths1 != paths2


# ---------------------------------------------------------------------------
# write_mix_file - output format
# ---------------------------------------------------------------------------


class TestWriteMixFile:
    def test_output_parseable(self, mix_file: Path, tmp_path: Path):
        """The output file should be parseable by the same parse_mix_file function."""
        entries = parse_mix_file(mix_file)
        groups = group_by_label(entries)
        counts = compute_sample_counts(groups, 10_500, TOTAL_TOKENS_SAMPLE)
        sampled = sample_submix(groups, counts, seed=42)

        out = tmp_path / "submix.txt"
        write_mix_file(sampled, out)

        reparsed = parse_mix_file(out)
        assert len(reparsed) == len(sampled)
        assert [e.path for e in reparsed] == [e.path for e in sampled]
        assert [e.label for e in reparsed] == [e.label for e in sampled]

    def test_output_has_section_headers(self, mix_file: Path, tmp_path: Path):
        entries = parse_mix_file(mix_file)
        groups = group_by_label(entries)
        counts = compute_sample_counts(groups, 10_500, TOTAL_TOKENS_SAMPLE)
        sampled = sample_submix(groups, counts, seed=42)

        out = tmp_path / "submix.txt"
        write_mix_file(sampled, out)

        content = out.read_text()
        assert "# FineMath-3Plus" in content
        assert "# Stack-Edu" in content

    def test_output_has_tokenizer_placeholder(self, mix_file: Path, tmp_path: Path):
        entries = parse_mix_file(mix_file)
        groups = group_by_label(entries)
        counts = compute_sample_counts(groups, 10_500, TOTAL_TOKENS_SAMPLE)
        sampled = sample_submix(groups, counts, seed=42)

        out = tmp_path / "submix.txt"
        write_mix_file(sampled, out)

        for entry in parse_mix_file(out):
            assert "{TOKENIZER}" in entry.path

    def test_creates_parent_dirs(self, tmp_path: Path):
        out = tmp_path / "nested" / "dir" / "submix.txt"
        write_mix_file([MixEntry("test", "path/{TOKENIZER}/file.npy")], out)
        assert out.exists()


# ---------------------------------------------------------------------------
# End-to-end: generate_submix
# ---------------------------------------------------------------------------


class TestGenerateSubmixEndToEnd:
    def test_roundtrip(self, mix_file: Path, tmp_path: Path):
        out = tmp_path / "sub.txt"
        summary = generate_submix(
            target_tokens=10_500,
            output_path=out,
            mix_file=mix_file,
            total_tokens=TOTAL_TOKENS_SAMPLE,
            seed=42,
        )

        expected_files = round(TOTAL_FILES_IN_SAMPLE * 10_500 / TOTAL_TOKENS_SAMPLE)
        assert out.exists()
        assert summary["sampled_files"] == expected_files
        assert summary["total_source_files"] == TOTAL_FILES_IN_SAMPLE

        # The output must be parse-compatible
        reparsed = parse_mix_file(out)
        assert len(reparsed) == expected_files

    def test_proportions_preserved_e2e(self, mix_file: Path, tmp_path: Path):
        out = tmp_path / "sub.txt"
        summary = generate_submix(
            target_tokens=10_500,
            output_path=out,
            mix_file=mix_file,
            total_tokens=TOTAL_TOKENS_SAMPLE,
            seed=42,
        )

        entries = parse_mix_file(mix_file)
        groups = group_by_label(entries)
        total_original = sum(len(g) for g in groups.values())

        for label, sampled_count in summary["labels"].items():
            orig_prop = len(groups[label]) / total_original
            samp_prop = sampled_count / summary["sampled_files"]
            diff = abs(orig_prop - samp_prop)
            assert diff < 0.10, (
                f"{label}: original={orig_prop:.4f}, sampled={samp_prop:.4f}"
            )

    def test_with_real_mix_3_8B(self, tmp_path: Path):
        """End-to-end test with the actual OLMo mix file for 3.8B tokens."""
        try:
            from scripts.generate_submix import get_default_mix_path

            mix_path = get_default_mix_path()
        except Exception:
            pytest.skip("OLMo mix file not available")

        out = tmp_path / "dolma3-3.8B.txt"
        summary = generate_submix(
            target_tokens=3.8e9,
            output_path=out,
            mix_file=mix_path,
            total_tokens=150e9,
            seed=42,
        )

        assert out.exists()
        assert summary["sampled_files"] > 0

        # Verify output is parseable
        reparsed = parse_mix_file(out)
        assert len(reparsed) == summary["sampled_files"]

        # Verify estimated tokens is in the right ballpark
        assert 3.0e9 < summary["estimated_tokens"] < 4.5e9

        # Verify all labels present
        original_entries = parse_mix_file(mix_path)
        original_labels = {e.label for e in original_entries}
        sampled_labels = {e.label for e in reparsed}
        assert sampled_labels.issubset(original_labels)
