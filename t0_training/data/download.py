"""Download and resolve npy data files from olmo-data.org for local training."""

import os
import sys
import time
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

REMOTE_BASE_URL = "https://olmo-data.org/"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MIX_FILE = str(_PROJECT_ROOT / "data" / "mixes" / "dolma3-3.8B.txt")
DEFAULT_DATA_DIR = str(_PROJECT_ROOT / "data" / "npy")


def parse_mix_file(mix_file: str, tokenizer_id: str) -> list[str]:
    """Parse a mix file and return relative paths (with tokenizer substituted)."""
    paths = []
    with open(mix_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            _label, path = line.split(",", 1)
            paths.append(path.replace("{TOKENIZER}", tokenizer_id))
    return paths


def resolve_data_paths(mix_file: str, data_dir: str, tokenizer_id: str) -> list[str]:
    """Resolve mix file entries to local paths, raising if any are missing."""
    rel_paths = parse_mix_file(mix_file, tokenizer_id)
    local_paths = []
    missing = []
    for rel_path in rel_paths:
        local_path = os.path.join(data_dir, rel_path)
        local_paths.append(local_path)
        if not os.path.exists(local_path):
            missing.append(rel_path)

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} data files missing from {data_dir}. "
            f"Run 't0-download --data-dir {data_dir} --mix-file {mix_file}' first.\n"
            f"First missing: {missing[0]}"
        )
    return local_paths


def _content_range_total(header: str | None) -> int | None:
    """Total size from a ``Content-Range`` header such as ``bytes */1234``, else None."""
    if not header:
        return None
    total = header.rsplit("/", 1)[-1].strip()
    return int(total) if total.isdigit() else None


def download_file(url: str, local_path: str, retries: int = 5) -> str:
    """Download a single file, retrying on transient errors.

    Data goes to ``<local_path>.part`` and is renamed only once complete and
    size-checked, so an interrupted download never leaves a truncated file at
    ``local_path``. Retries resume from the partial file via HTTP Range.
    """
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    part_path = local_path + ".part"
    last_err: Exception | None = None
    for attempt in range(retries):
        if attempt:
            time.sleep(min(2**attempt, 30))
        try:
            offset = os.path.getsize(part_path) if os.path.exists(part_path) else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with requests.get(url, stream=True, timeout=(10, 60), headers=headers) as resp:
                if resp.status_code == 416:
                    # Offset not satisfiable. The .part file is complete only if its size
                    # equals the total the server reports ("Content-Range: bytes */N");
                    # otherwise (too big, or total unknown) discard it and retry from zero.
                    total = _content_range_total(resp.headers.get("Content-Range"))
                    if total is not None and total == offset:
                        os.replace(part_path, local_path)
                        return local_path
                    os.remove(part_path)
                    last_err = IOError(f"unsatisfiable range for {offset}-byte partial file (server total: {total})")
                    continue
                resp.raise_for_status()
                if offset and resp.status_code != 206:
                    offset = 0  # server ignored Range; restart from scratch
                expected = int(resp.headers.get("Content-Length", -1))
                with open(part_path, "ab" if offset else "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
                if expected >= 0 and os.path.getsize(part_path) != offset + expected:
                    raise IOError("size mismatch after download")
            os.replace(part_path, local_path)
            return local_path
        except (requests.RequestException, OSError) as e:
            last_err = e
    raise RuntimeError(f"giving up after {retries} attempts: {last_err}")


@dataclass
class VerifyResult:
    """Outcome of comparing local files with the server's sizes."""

    missing: list[str] = field(default_factory=list)  # not on disk
    mismatched: list[str] = field(default_factory=list)  # confirmed wrong size
    unverified: list[str] = field(default_factory=list)  # server check failed; file may be fine

    @property
    def ok(self) -> bool:
        return not (self.missing or self.mismatched or self.unverified)


def verify_paths(rel_paths: list[str], data_dir: str, workers: int = 16) -> VerifyResult:
    """Compare each local file's size to the server's Content-Length.

    A file is only reported as ``mismatched`` when the server answered with a size that
    differs. If the server could not be queried, it is ``unverified`` (not known to be bad).
    """

    def check(rel_path: str) -> tuple[str, str]:
        local = os.path.join(data_dir, rel_path)
        if not os.path.exists(local):
            return "missing", local
        for attempt in range(3):
            try:
                r = requests.head(REMOTE_BASE_URL + rel_path, allow_redirects=True, timeout=30)
                r.raise_for_status()
                want = int(r.headers["Content-Length"])
                return ("ok" if want == os.path.getsize(local) else "mismatched"), local
            except (requests.RequestException, KeyError, ValueError):
                time.sleep(2**attempt)
        return "unverified", local

    result = VerifyResult()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for status, local in pool.map(check, rel_paths):
            if status != "ok":
                getattr(result, status).append(local)
    return result


def verify_mix(mix_file: str, data_dir: str, tokenizer_id: str, workers: int = 16) -> VerifyResult:
    return verify_paths(parse_mix_file(mix_file, tokenizer_id), data_dir, workers)


def eval_data_paths() -> list[str]:
    """Relative paths of the LM-evaluator (perplexity validation) shards.

    Uses the same olmo-core mix as the ``lm_evaluator`` callback, so the files
    land where ``mix_base_dir: <data_dir>`` expects them.
    """
    from olmo_core.data import DataMix, TokenizerConfig

    tok = TokenizerConfig.dolma2()
    urls, _labels = DataMix.v3_small_ppl_validation.build(REMOTE_BASE_URL, tok.identifier)
    return [u[len(REMOTE_BASE_URL) :] if u.startswith(REMOTE_BASE_URL) else u.lstrip("/") for u in urls]


def _report_verification(res: VerifyResult) -> None:
    def show(title: str, paths: list[str]) -> None:
        print(f"\n{len(paths)} {title}", file=sys.stderr)
        for p in paths[:20]:
            print(f"  {p}", file=sys.stderr)
        if len(paths) > 20:
            print(f"  ... and {len(paths) - 20} more", file=sys.stderr)

    if res.mismatched:
        show("files have the wrong size (truncated or corrupt). Delete them and re-run:", res.mismatched)
    if res.missing:
        show("files are missing. Re-run to download them:", res.missing)
    if res.unverified:
        show(
            "files could not be verified (server did not answer). They are NOT known to be bad:"
            " do not delete them; re-run later to verify.",
            res.unverified,
        )


def _sync(rel_paths: list[str], data_dir: str, workers: int, verify: bool) -> list[str]:
    to_download = []
    for rel_path in rel_paths:
        local_path = os.path.join(data_dir, rel_path)
        if not os.path.exists(local_path):
            to_download.append((REMOTE_BASE_URL + rel_path, local_path))

    if not to_download:
        print(f"All {len(rel_paths)} files already present in {data_dir}")
    else:
        print(f"Downloading {len(to_download)}/{len(rel_paths)} files to {data_dir}")
        failed = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(download_file, url, path): (url, path) for url, path in to_download}
            for i, future in enumerate(as_completed(futures), 1):
                url, path = futures[future]
                try:
                    future.result()
                    print(f"  [{i}/{len(to_download)}] {os.path.basename(path)}")
                except Exception as e:
                    print(f"  [{i}/{len(to_download)}] FAILED {url}: {e}", file=sys.stderr)
                    failed.append(url)
        if failed:
            print(f"\n{len(failed)} downloads failed.", file=sys.stderr)
            sys.exit(1)

    if verify:
        print("Verifying file sizes against server...")
        res = verify_paths(rel_paths, data_dir)
        if not res.ok:
            _report_verification(res)
            sys.exit(1)
        print("All files verified.")

    print("Done.")
    return [os.path.join(data_dir, p) for p in rel_paths]


def download_mix(
    mix_file: str, data_dir: str, tokenizer_id: str, workers: int = 8, verify: bool = True
) -> list[str]:
    """Download all files in a mix that aren't already present locally.

    With ``verify`` (default), afterwards checks every file's size against the
    server and exits non-zero listing any mismatches. Delete those and re-run.

    Returns the list of local paths (all files, not just newly downloaded).
    """
    return _sync(parse_mix_file(mix_file, tokenizer_id), data_dir, workers, verify)


def download_eval_data(data_dir: str, workers: int = 8, verify: bool = True) -> list[str]:
    """Download the LM-evaluator validation shards into ``data_dir``."""
    return _sync(eval_data_paths(), data_dir, workers, verify)
