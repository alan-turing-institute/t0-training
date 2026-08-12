"""Tests for t0_training/generate_submix.py"""

from collections import OrderedDict
from pathlib import Path

import pytest

from t0_training.olmo.generate_submix import (
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
    # Verifies that all data lines are parsed and comments/blanks are skipped.
    def test_parses_correct_count(self, mix_file: Path):
        entries = parse_mix_file(mix_file)
        assert len(entries) == TOTAL_FILES_IN_SAMPLE

    # Verifies the {TOKENIZER} placeholder is preserved (not resolved at this stage).
    def test_all_paths_have_tokenizer_placeholder(self, mix_file: Path):
        entries = parse_mix_file(mix_file)
        for entry in entries:
            assert "{TOKENIZER}" in entry.path

    # Verifies all expected source labels are extracted from the mix.
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
    # Verifies entries are correctly grouped and each group has the right file count.
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


# ---------------------------------------------------------------------------
# compute_sample_counts - proportional allocation
# ---------------------------------------------------------------------------


class TestComputeSampleCounts:
    # Requesting half the tokens should yield half the total files.
    def test_half_gives_roughly_half_per_label(self, mix_file: Path):
        entries = parse_mix_file(mix_file)
        groups = group_by_label(entries)
        counts = compute_sample_counts(groups, 10_500, TOTAL_TOKENS_SAMPLE)
        total_sampled = sum(counts.values())
        assert total_sampled == round(TOTAL_FILES_IN_SAMPLE * 0.5)

    # Every source label must get at least 1 file to preserve the mix composition.
    def test_minimum_one_per_label(self, mix_file: Path):
        entries = parse_mix_file(mix_file)
        groups = group_by_label(entries)
        counts = compute_sample_counts(groups, 9_000, TOTAL_TOKENS_SAMPLE)
        for label, count in counts.items():
            assert count >= 1, f"{label} got {count} files"

    # A label should never get more files than it has in the original mix.
    def test_no_label_exceeds_original_count(self, mix_file: Path):
        entries = parse_mix_file(mix_file)
        groups = group_by_label(entries)
        counts = compute_sample_counts(groups, 20_000, TOTAL_TOKENS_SAMPLE)
        for label, count in counts.items():
            assert count <= len(groups[label]), (
                f"{label}: sampled {count} > available {len(groups[label])}"
            )

    # The total sampled file count should match the expected count for several target sizes.
    def test_total_matches_target_file_count(self, mix_file: Path):
        entries = parse_mix_file(mix_file)
        groups = group_by_label(entries)
        for target_tokens in [9_000, 12_000, 15_000, 20_000]:
            counts = compute_sample_counts(groups, target_tokens, TOTAL_TOKENS_SAMPLE)
            total_sampled = sum(counts.values())
            expected_files = round(TOTAL_FILES_IN_SAMPLE * target_tokens / TOTAL_TOKENS_SAMPLE)
            assert total_sampled == expected_files, (
                f"target={target_tokens}: got {total_sampled} files, expected {expected_files}"
            )


# ---------------------------------------------------------------------------
# Proportionality — the core property of the sub-mix
# ---------------------------------------------------------------------------


class TestProportionality:
    # Label proportions in the sub-mix should be within 10% of the original mix.
    def test_proportions_close_at_50pct(self, mix_file: Path):
        entries = parse_mix_file(mix_file)
        groups = group_by_label(entries)
        counts = compute_sample_counts(groups, 10_500, TOTAL_TOKENS_SAMPLE)

        total_original = sum(len(g) for g in groups.values())
        total_sampled = sum(counts.values())

        for label in groups:
            orig_prop = len(groups[label]) / total_original
            samp_prop = counts[label] / total_sampled
            diff = abs(orig_prop - samp_prop)
            assert diff < 0.10, (
                f"{label}: original={orig_prop:.4f}, "
                f"sampled={samp_prop:.4f}, diff={diff:.4f}"
            )

    # Same check on the real OLMo 150B mix at the 3.8B target we actually use.
    def test_proportions_close_on_real_mix(self):
        try:
            from t0_training.olmo.generate_submix import get_default_mix_path

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
            assert diff < 0.01, (
                f"{label}: original={orig_prop:.4f}, sampled={samp_prop:.4f}, diff={diff:.4f}"
            )


# ---------------------------------------------------------------------------
# sample_submix
# ---------------------------------------------------------------------------


class TestSampleSubmix:
    # Each label gets exactly the number of files specified by compute_sample_counts.
    def test_correct_count_per_label(self, mix_file: Path):
        entries = parse_mix_file(mix_file)
        groups = group_by_label(entries)
        counts = compute_sample_counts(groups, 10_500, TOTAL_TOKENS_SAMPLE)
        sampled = sample_submix(groups, counts, seed=42)

        sampled_groups = group_by_label(sampled)
        for label, count in counts.items():
            assert len(sampled_groups.get(label, [])) == count

    # No file should appear twice in the sampled sub-mix.
    def test_no_duplicates(self, mix_file: Path):
        entries = parse_mix_file(mix_file)
        groups = group_by_label(entries)
        counts = compute_sample_counts(groups, 10_500, TOTAL_TOKENS_SAMPLE)
        sampled = sample_submix(groups, counts, seed=42)

        paths = [e.path for e in sampled]
        assert len(paths) == len(set(paths)), "Duplicate paths in sub-mix"

    # Same seed must produce the same sub-mix (reproducibility for experiments).
    def test_deterministic_with_same_seed(self, mix_file: Path):
        entries = parse_mix_file(mix_file)
        groups = group_by_label(entries)
        counts = compute_sample_counts(groups, 10_500, TOTAL_TOKENS_SAMPLE)

        sampled1 = sample_submix(groups, counts, seed=123)
        sampled2 = sample_submix(groups, counts, seed=123)
        assert [e.path for e in sampled1] == [e.path for e in sampled2]


# ---------------------------------------------------------------------------
# write_mix_file - output format
# ---------------------------------------------------------------------------


class TestWriteMixFile:
    # The written mix file must be parseable back into the same entries (roundtrip).
    def test_output_parseable(self, mix_file: Path, tmp_path: Path):
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


# ---------------------------------------------------------------------------
# End-to-end: generate_submix
# ---------------------------------------------------------------------------


class TestGenerateSubmixEndToEnd:
    # Full pipeline: generate a sub-mix file and verify it has the right number of files
    # and is parseable.
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

        reparsed = parse_mix_file(out)
        assert len(reparsed) == expected_files

    # End-to-end with the actual OLMo 150B mix at 3.8B target (our default training size).
    def test_with_real_mix_3_8B(self, tmp_path: Path):
        try:
            from t0_training.olmo.generate_submix import get_default_mix_path

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

        reparsed = parse_mix_file(out)
        assert len(reparsed) == summary["sampled_files"]
        assert 3.0e9 < summary["estimated_tokens"] < 4.5e9
