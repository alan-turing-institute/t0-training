"""Download npy data files from olmo-data.org for local training."""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

REMOTE_BASE_URL = "https://olmo-data.org/"
DEFAULT_MIX_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "mixes", "dolma3-3.8B.txt")
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "npy")


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


def download_file(url: str, local_path: str) -> str:
    """Download a single file. Returns the local path on success."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    resp = requests.get(url, stream=True, timeout=(10, 60))
    resp.raise_for_status()
    with open(local_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
    return local_path


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


def main():
    parser = argparse.ArgumentParser(description="Download npy data files for training.")
    parser.add_argument("--mix-file", default=DEFAULT_MIX_FILE, help="Path to mix file.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Local directory to store files.")
    parser.add_argument("--tokenizer-id", default="allenai/dolma2-tokenizer", help="Tokenizer identifier.")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel downloads.")
    args = parser.parse_args()

    download_mix(args.mix_file, args.data_dir, args.tokenizer_id, args.workers)


if __name__ == "__main__":
    main()
