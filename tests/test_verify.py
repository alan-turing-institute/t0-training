"""Tests for size verification in t0_training.data.download."""

from unittest.mock import MagicMock, patch

import pytest

from t0_training.data import download
from t0_training.data.download import _report_verification, verify_paths


@pytest.fixture(autouse=True)
def no_sleep():
    with patch.object(download.time, "sleep"):
        yield


def head_ok(size):
    r = MagicMock()
    r.headers = {"Content-Length": str(size)}
    return r


def write(tmp_path, rel, n):
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x" * n)


def test_matching_size_is_ok(tmp_path):
    write(tmp_path, "a.npy", 10)
    with patch.object(download.requests, "head", return_value=head_ok(10)):
        res = verify_paths(["a.npy"], str(tmp_path))
    assert res.ok


def test_confirmed_mismatch(tmp_path):
    write(tmp_path, "a.npy", 5)
    with patch.object(download.requests, "head", return_value=head_ok(10)):
        res = verify_paths(["a.npy"], str(tmp_path))
    assert res.mismatched == [str(tmp_path / "a.npy")]
    assert not res.missing and not res.unverified


def test_missing_file(tmp_path):
    with patch.object(download.requests, "head", return_value=head_ok(10)) as head:
        res = verify_paths(["gone.npy"], str(tmp_path))
    assert res.missing == [str(tmp_path / "gone.npy")]
    head.assert_not_called()


def test_server_failure_is_unverified_not_mismatched(tmp_path):
    write(tmp_path, "a.npy", 10)
    with patch.object(download.requests, "head", side_effect=download.requests.ConnectionError("down")):
        res = verify_paths(["a.npy"], str(tmp_path))
    assert res.unverified == [str(tmp_path / "a.npy")]
    assert not res.mismatched and not res.missing
    assert not res.ok


def test_http_error_is_unverified(tmp_path):
    write(tmp_path, "a.npy", 10)
    bad = MagicMock()
    bad.raise_for_status.side_effect = download.requests.HTTPError("503")
    with patch.object(download.requests, "head", return_value=bad):
        res = verify_paths(["a.npy"], str(tmp_path))
    assert res.unverified and not res.mismatched


def test_missing_content_length_is_unverified(tmp_path):
    write(tmp_path, "a.npy", 10)
    r = MagicMock()
    r.headers = {}
    with patch.object(download.requests, "head", return_value=r):
        res = verify_paths(["a.npy"], str(tmp_path))
    assert res.unverified and not res.mismatched


def test_report_only_recommends_deleting_confirmed_bad(tmp_path, capsys):
    res = download.VerifyResult(mismatched=["bad.npy"], unverified=["maybe.npy"], missing=["gone.npy"])
    _report_verification(res)
    err = capsys.readouterr().err
    assert "Delete them" in err
    delete_section = err.split("Delete them and re-run:")[1].split("\n\n")[0]
    assert "bad.npy" in delete_section
    assert "maybe.npy" not in delete_section
    assert "do not delete" in err
