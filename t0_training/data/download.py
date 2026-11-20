"""Download and resolve npy data files from olmo-data.org for local training."""

import os
import sys
import time
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
                if resp.status_code == 416:  # .part already holds the full file
                    offset = 0
                    resp.close()
                    os.replace(part_path, local_path)
                    return local_path
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


def download_mix(mix_file: str, data_dir: str, tokenizer_id: str, workers: int = 8) -> list[str]:
    """Download all files in a mix that aren't already present locally.

    Returns the list of local paths (all files, not just newly downloaded).
    """
    rel_paths = parse_mix_file(mix_file, tokenizer_id)

    to_download = []
    for rel_path in rel_paths:
        local_path = os.path.join(data_dir, rel_path)
        if not os.path.exists(local_path):
            to_download.append((REMOTE_BASE_URL + rel_path, local_path))

    if not to_download:
        print(f"All {len(rel_paths)} files already present in {data_dir}")
        return [os.path.join(data_dir, p) for p in rel_paths]

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

    print("Done.")
    return [os.path.join(data_dir, p) for p in rel_paths]
