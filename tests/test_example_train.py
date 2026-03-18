"""Tests for mix file parsing and data downloading."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from scripts.download_data import parse_mix_file, download_file, download_mix, REMOTE_BASE_URL

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
    def test_correct_count(self, mix_file: Path):
        paths = parse_mix_file(str(mix_file), TOKENIZER_ID)
        assert len(paths) == 4

    def test_skips_comments_and_blanks(self, mix_file: Path):
        paths = parse_mix_file(str(mix_file), TOKENIZER_ID)
        for p in paths:
            assert not p.endswith(".txt")
            assert "#" not in p

    def test_tokenizer_resolved(self, mix_file: Path):
        paths = parse_mix_file(str(mix_file), TOKENIZER_ID)
        for p in paths:
            assert "{TOKENIZER}" not in p
            assert TOKENIZER_ID in p

    def test_returns_relative_paths(self, mix_file: Path):
        paths = parse_mix_file(str(mix_file), TOKENIZER_ID)
        for p in paths:
            assert not p.startswith("/")
            assert not p.startswith("http")

    def test_full_path_format(self, mix_file: Path):
        paths = parse_mix_file(str(mix_file), TOKENIZER_ID)
        assert paths[0] == (
            "preprocessed/dolma2-0625/v0.1-150b/"
            "dolma2-tokenizer/finemath-3plus/part-000-00000.npy"
        )

    def test_all_paths_are_npy(self, mix_file: Path):
        paths = parse_mix_file(str(mix_file), TOKENIZER_ID)
        for p in paths:
            assert p.endswith(".npy")

    def test_empty_file(self, tmp_path: Path):
        empty = tmp_path / "empty.txt"
        empty.write_text("")
        assert parse_mix_file(str(empty), TOKENIZER_ID) == []

    def test_comments_only(self, tmp_path: Path):
        f = tmp_path / "comments.txt"
        f.write_text("# just a comment\n# another\n")
        assert parse_mix_file(str(f), TOKENIZER_ID) == []

    def test_different_tokenizer(self, mix_file: Path):
        paths = parse_mix_file(str(mix_file), "my-custom-tokenizer")
        for p in paths:
            assert "my-custom-tokenizer" in p
            assert "dolma2-tokenizer" not in p


class TestDownloadFile:
    def test_downloads_to_local_path(self, tmp_path: Path):
        local_path = str(tmp_path / "subdir" / "file.npy")
        fake_content = b"fake npy data"

        mock_resp = MagicMock()
        mock_resp.iter_content.return_value = [fake_content]

        with patch("scripts.download_data.requests.get", return_value=mock_resp) as mock_get:
            result = download_file("https://example.com/file.npy", local_path)

        assert result == local_path
        assert Path(local_path).read_bytes() == fake_content
        mock_get.assert_called_once_with("https://example.com/file.npy", stream=True, timeout=(10, 60))
        mock_resp.raise_for_status.assert_called_once()

    def test_creates_parent_directories(self, tmp_path: Path):
        local_path = str(tmp_path / "a" / "b" / "c" / "file.npy")
        mock_resp = MagicMock()
        mock_resp.iter_content.return_value = [b"data"]

        with patch("scripts.download_data.requests.get", return_value=mock_resp):
            download_file("https://example.com/file.npy", local_path)

        assert Path(local_path).exists()


class TestDownloadMix:
    def test_skips_existing_files(self, tmp_path: Path, mix_file: Path):
        data_dir = tmp_path / "data"
        # Pre-create all files
        for rel_path in parse_mix_file(str(mix_file), TOKENIZER_ID):
            p = data_dir / rel_path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"existing")

        with patch("scripts.download_data.download_file") as mock_dl:
            paths = download_mix(str(mix_file), str(data_dir), TOKENIZER_ID)

        mock_dl.assert_not_called()
        assert len(paths) == 4
        for p in paths:
            assert p.startswith(str(data_dir))

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

        with patch("scripts.download_data.download_file", side_effect=fake_download) as mock_dl:
            paths = download_mix(str(mix_file), str(data_dir), TOKENIZER_ID, workers=2)

        assert mock_dl.call_count == 3
        assert len(paths) == 4
        # Verify URLs were correct
        called_urls = {call.args[0] for call in mock_dl.call_args_list}
        for rel_path in rel_paths[1:]:
            assert REMOTE_BASE_URL + rel_path in called_urls

    def test_returns_local_paths(self, tmp_path: Path, mix_file: Path):
        data_dir = tmp_path / "data"

        def fake_download(url, local_path):
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            Path(local_path).write_bytes(b"data")
            return local_path

        with patch("scripts.download_data.download_file", side_effect=fake_download):
            paths = download_mix(str(mix_file), str(data_dir), TOKENIZER_ID, workers=2)

        for p in paths:
            assert p.startswith(str(data_dir))
            assert p.endswith(".npy")

    def test_exits_on_failure(self, tmp_path: Path, mix_file: Path):
        data_dir = tmp_path / "data"

        with patch("scripts.download_data.download_file", side_effect=Exception("network error")):
            with pytest.raises(SystemExit):
                download_mix(str(mix_file), str(data_dir), TOKENIZER_ID, workers=2)

    def test_empty_mix_returns_empty(self, tmp_path: Path):
        empty_mix = tmp_path / "empty.txt"
        empty_mix.write_text("")
        data_dir = tmp_path / "data"

        with patch("scripts.download_data.download_file") as mock_dl:
            paths = download_mix(str(empty_mix), str(data_dir), TOKENIZER_ID)

        mock_dl.assert_not_called()
        assert paths == []
