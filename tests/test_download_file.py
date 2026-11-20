"""Tests for resumable downloads in t0_training.data.download (HTTP 416 handling)."""

from unittest.mock import patch

import pytest

from t0_training.data import download
from t0_training.data.download import _content_range_total, download_file

URL = "https://example.invalid/part-000.npy"


class FakeResponse:
    def __init__(self, status_code=200, body=b"", headers=None):
        self.status_code = status_code
        self._body = body
        self.headers = dict(headers or {})
        if status_code in (200, 206):
            self.headers.setdefault("Content-Length", str(len(body)))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise download.requests.HTTPError(f"{self.status_code}")

    def iter_content(self, chunk_size=1):
        yield self._body


@pytest.fixture(autouse=True)
def no_sleep():
    with patch.object(download.time, "sleep"):
        yield


def test_content_range_total():
    assert _content_range_total("bytes */1234") == 1234
    assert _content_range_total("bytes 0-9/1234") == 1234
    assert _content_range_total("bytes */*") is None
    assert _content_range_total("") is None
    assert _content_range_total(None) is None


def test_416_with_complete_part_file_is_promoted(tmp_path):
    dest = tmp_path / "f.npy"
    (tmp_path / "f.npy.part").write_bytes(b"x" * 10)
    with patch.object(download.requests, "get", return_value=FakeResponse(416, headers={"Content-Range": "bytes */10"})):
        download_file(URL, str(dest))
    assert dest.read_bytes() == b"x" * 10
    assert not (tmp_path / "f.npy.part").exists()


def test_416_with_oversized_part_file_is_discarded_and_redownloaded(tmp_path):
    dest = tmp_path / "f.npy"
    (tmp_path / "f.npy.part").write_bytes(b"x" * 50)  # bigger than the real file
    responses = [
        FakeResponse(416, headers={"Content-Range": "bytes */10"}),
        FakeResponse(200, body=b"y" * 10),
    ]
    with patch.object(download.requests, "get", side_effect=responses) as get:
        download_file(URL, str(dest))
    assert dest.read_bytes() == b"y" * 10  # the correct file, not the oversized one
    assert "Range" not in get.call_args_list[1].kwargs["headers"]  # restarted from zero


def test_416_without_content_range_is_not_trusted(tmp_path):
    dest = tmp_path / "f.npy"
    (tmp_path / "f.npy.part").write_bytes(b"x" * 10)
    responses = [FakeResponse(416), FakeResponse(200, body=b"y" * 7)]
    with patch.object(download.requests, "get", side_effect=responses):
        download_file(URL, str(dest))
    assert dest.read_bytes() == b"y" * 7


def test_persistent_416_never_creates_final_file(tmp_path):
    dest = tmp_path / "f.npy"
    with patch.object(download.requests, "get", return_value=FakeResponse(416, headers={"Content-Range": "bytes */5"})):
        (tmp_path / "f.npy.part").write_bytes(b"x" * 10)
        # part file is removed on the first 416, so later attempts send no Range and see 416 again
        with pytest.raises(RuntimeError):
            download_file(URL, str(dest), retries=3)
    assert not dest.exists()


def test_resume_with_206_appends(tmp_path):
    dest = tmp_path / "f.npy"
    (tmp_path / "f.npy.part").write_bytes(b"abc")
    resp = FakeResponse(206, body=b"defg", headers={"Content-Range": "bytes 3-6/7"})
    with patch.object(download.requests, "get", return_value=resp):
        download_file(URL, str(dest))
    assert dest.read_bytes() == b"abcdefg"


def test_short_206_range_is_not_promoted(tmp_path):
    """Content-Range total (10) must be reached; a short range must keep the .part and retry."""
    dest = tmp_path / "f.npy"
    (tmp_path / "f.npy.part").write_bytes(b"abc")
    responses = [
        FakeResponse(206, body=b"def", headers={"Content-Range": "bytes 3-5/10"}),  # only 6 of 10
        FakeResponse(206, body=b"ghij", headers={"Content-Range": "bytes 6-9/10"}),
    ]
    with patch.object(download.requests, "get", side_effect=responses) as get:
        download_file(URL, str(dest))
    assert dest.read_bytes() == b"abcdefghij"
    assert get.call_args_list[1].kwargs["headers"]["Range"] == "bytes=6-"  # resumed, not restarted


def test_short_206_never_promoted_when_server_keeps_truncating(tmp_path):
    dest = tmp_path / "f.npy"
    (tmp_path / "f.npy.part").write_bytes(b"abc")
    resp = lambda: FakeResponse(206, body=b"", headers={"Content-Range": "bytes 3-5/10", "Content-Length": "0"})
    with patch.object(download.requests, "get", side_effect=lambda *a, **k: resp()):
        with pytest.raises(RuntimeError):
            download_file(URL, str(dest), retries=3)
    assert not dest.exists()


def test_missing_total_size_is_not_promoted(tmp_path):
    dest = tmp_path / "f.npy"
    resp = FakeResponse(200, body=b"abc")
    del resp.headers["Content-Length"]
    with patch.object(download.requests, "get", return_value=resp):
        with pytest.raises(RuntimeError):
            download_file(URL, str(dest), retries=2)
    assert not dest.exists()
