"""Tests for t0_training.olmo.data — mix file parsing and data downloading."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from t0_training.olmo.data import (
    parse_mix_file,
    download_file,
    download_mix,
    resolve_data_paths,
    REMOTE_BASE_URL,
)

SAMPLE_MIX = """\
# FineMath-3Plus
finemath-3plus,preprocessed/dolma2-0625/v0.1-150b/{TOKENIZER}/finemath-3plus/part-000-00000.npy
finemath-3plus,preprocessed/dolma2-0625/v0.1-150b/{TOKENIZER}/finemath-3plus/part-001-00000.npy
# Arxiv
arxiv,preprocessed/dolma2-0625/v0.1-150b/{TOKENIZER}/arxiv/part-000-00000.npy

# Stack-Edu
stack-edu_Python,preprocessed/dolma2-0625/v0.1-150b/{TOKENIZER}/stack-edu/Python/part-000-00000.npy
"""

TOKENIZER_ID = "dolma2-tokenizer"


@pytest.fixture
def mix_file(tmp_path: Path) -> Path:
    p = tmp_path / "test-mix.txt"
    p.write_text(SAMPLE_MIX)
    return p


class TestParseMixFile:
    # Verifies that comments and blank lines are skipped, only data lines are parsed.
    def test_correct_count_skipping_comments_and_blanks(self, mix_file: Path):
        paths = parse_mix_file(str(mix_file), TOKENIZER_ID)
        assert len(paths) == 4

    # Verifies the {TOKENIZER} placeholder is replaced with the actual tokenizer id.
    def test_tokenizer_resolved(self, mix_file: Path):
        paths = parse_mix_file(str(mix_file), TOKENIZER_ID)
        for p in paths:
            assert "{TOKENIZER}" not in p
            assert TOKENIZER_ID in p

    # Verifies the exact output format: label stripped, tokenizer substituted, relative path.
    def test_full_path_format(self, mix_file: Path):
        paths = parse_mix_file(str(mix_file), TOKENIZER_ID)
        assert paths[0] == (
            "preprocessed/dolma2-0625/v0.1-150b/"
            "dolma2-tokenizer/finemath-3plus/part-000-00000.npy"
        )


class TestResolveDataPaths:
    # When all npy files exist locally, resolve_data_paths returns their absolute paths.
    def test_all_present(self, tmp_path: Path, mix_file: Path):
        data_dir = tmp_path / "data"
        for rel_path in parse_mix_file(str(mix_file), TOKENIZER_ID):
            p = data_dir / rel_path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"data")

        paths = resolve_data_paths(str(mix_file), str(data_dir), TOKENIZER_ID)
        assert len(paths) == 4
        for p in paths:
            assert p.startswith(str(data_dir))

    # When npy files are missing, a clear error tells the user to download first.
    def test_missing_raises(self, tmp_path: Path, mix_file: Path):
        data_dir = tmp_path / "empty_data"
        with pytest.raises(FileNotFoundError, match="data files missing"):
            resolve_data_paths(str(mix_file), str(data_dir), TOKENIZER_ID)


class TestDownloadFile:
    # Verifies download_file fetches from the URL and writes content to the local path.
    def test_downloads_to_local_path(self, tmp_path: Path):
        local_path = str(tmp_path / "subdir" / "file.npy")
        fake_content = b"fake npy data"

        mock_resp = MagicMock()
        mock_resp.iter_content.return_value = [fake_content]

        with patch("t0_training.olmo.data.requests.get", return_value=mock_resp) as mock_get:
            result = download_file("https://example.com/file.npy", local_path)

        assert result == local_path
        assert Path(local_path).read_bytes() == fake_content
        mock_get.assert_called_once_with("https://example.com/file.npy", stream=True, timeout=(10, 60))
        mock_resp.raise_for_status.assert_called_once()


class TestDownloadMix:
    # Files already on disk should not be re-downloaded.
    def test_skips_existing_files(self, tmp_path: Path, mix_file: Path):
        data_dir = tmp_path / "data"
        for rel_path in parse_mix_file(str(mix_file), TOKENIZER_ID):
            p = data_dir / rel_path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"existing")

        with patch("t0_training.olmo.data.download_file") as mock_dl:
            paths = download_mix(str(mix_file), str(data_dir), TOKENIZER_ID)

        mock_dl.assert_not_called()
        assert len(paths) == 4

    # Only missing files are downloaded; existing ones are kept. URLs are correctly constructed.
    def test_downloads_missing_files(self, tmp_path: Path, mix_file: Path):
        data_dir = tmp_path / "data"
        rel_paths = parse_mix_file(str(mix_file), TOKENIZER_ID)

        # Pre-create only the first file
        first = data_dir / rel_paths[0]
        first.parent.mkdir(parents=True, exist_ok=True)
        first.write_bytes(b"existing")

        def fake_download(url, local_path):
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            Path(local_path).write_bytes(b"downloaded")
            return local_path

        with patch("t0_training.olmo.data.download_file", side_effect=fake_download) as mock_dl:
            paths = download_mix(str(mix_file), str(data_dir), TOKENIZER_ID, workers=2)

        assert mock_dl.call_count == 3
        assert len(paths) == 4
        called_urls = {call.args[0] for call in mock_dl.call_args_list}
        for rel_path in rel_paths[1:]:
            assert REMOTE_BASE_URL + rel_path in called_urls

    # A download failure should cause a non-zero exit so the training script doesn't proceed.
    def test_exits_on_failure(self, tmp_path: Path, mix_file: Path):
        data_dir = tmp_path / "data"

        with patch("t0_training.olmo.data.download_file", side_effect=Exception("network error")):
            with pytest.raises(SystemExit):
                download_mix(str(mix_file), str(data_dir), TOKENIZER_ID, workers=2)
