#!/usr/bin/env python3
"""Tokenize a sample of the pure `allenai/dolma3_longmino_pool` dataset into OLMo-core's
pretraining npy shard format.

Unlike `OLMo-longmino-mix-0625.txt` (OLMo3's own already-blended 33.9%-long/66.1%-midtrain
mix — see planning/long_context_extension_plan.md's 2026-11-10 correction), this pool is raw,
100% genuine long documents, but isn't pre-tokenized — this script does that tokenization step,
which previously didn't exist anywhere in this pipeline.

Produces token_ids_part_NNNN.npy files: raw headerless uint32 binary (no npy header), same
convention as t0_training/olmo/convert_sft_data.py's SFT shards and every other npy file this
pipeline reads via olmo_core's byte-offset memmap loading. Documents are concatenated with an
EOS separator between them, chunked every `--chunk-tokens` tokens (default 250M, ~1GB at
uint32, matching convert_sft_data.py's chunk size).

Usage:
  uv run --no-sync python scripts/tokenize_longmino_pool.py \\
      --target-tokens 7e9 --output-dir data/npy/longmino-pool-tokenized
"""
import argparse
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_DATASET = "allenai/dolma3_longmino_pool"
# Matches the identifier the upstream longmino shards are hosted under (see the `sed`
# substitution in docs/5_long_context_extension_2048_8192.md) — same underlying vocab as
# TokenizerConfig.dolma2() (vocab_size=100278), just published under this HF repo name.
DEFAULT_TOKENIZER = "allenai/dolma3-tokenizer"
CHUNK_TOKENS_DEFAULT = 250_000_000  # ~1 GB per chunk at uint32, matches convert_sft_data.py


def _write_chunk(out_dir: Path, chunk_idx: int, ids: list) -> int:
    arr = np.array(ids, dtype=np.uint32)
    part = f"{chunk_idx:04d}"
    arr.tofile(out_dir / f"token_ids_part_{part}.npy")
    log.info("Wrote chunk %s: %d tokens", part, len(arr))
    return len(arr)


def tokenize_pool(
    dataset: str,
    tokenizer_id: str,
    target_tokens: int,
    output_dir: Path,
    chunk_tokens: int,
    seed: int,
    text_field: str,
) -> None:
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from tqdm import tqdm

    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError(f"{tokenizer_id} has no eos_token_id — can't separate documents")

    log.info("Streaming %s (shuffled, seed=%d)...", dataset, seed)
    ds = load_dataset(dataset, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    all_ids: list = []
    chunk_idx = 0
    total_tokens = 0
    n_docs = 0
    n_skipped = 0

    pbar = tqdm(total=target_tokens, unit="tok", desc="Tokenizing")
    for ex in ds:
        if total_tokens >= target_tokens:
            break
        text = ex.get(text_field)
        if not text:
            n_skipped += 1
            continue

        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            n_skipped += 1
            continue

        if all_ids:
            all_ids.append(eos_id)
        all_ids.extend(ids)
        n_docs += 1

        if len(all_ids) >= chunk_tokens:
            written = _write_chunk(output_dir, chunk_idx, all_ids[:chunk_tokens])
            total_tokens += written
            pbar.update(written)
            # carry any overflow into the next chunk rather than dropping it
            all_ids = all_ids[chunk_tokens:]
            chunk_idx += 1

    if all_ids:
        written = _write_chunk(output_dir, chunk_idx, all_ids)
        total_tokens += written
        pbar.update(written)
        chunk_idx += 1
    pbar.close()

    log.info(
        "Done. %d documents tokenized (%d skipped), %d tokens across %d chunks, written to %s",
        n_docs, n_skipped, total_tokens, chunk_idx, output_dir,
    )

    # Emit a ready-to-use mix-file fragment: one line per part file, shared label.
    # Append this (or a prefix of it, for a smaller token budget) into your combined mix file
    # alongside a t0-submix-generated short-context sample — see the plan for exact commands.
    mix_fragment_path = output_dir / "mix_fragment.txt"
    rel_prefix = output_dir.as_posix()
    with open(mix_fragment_path, "w") as f:
        for i in range(chunk_idx):
            f.write(f"longmino-pool,{rel_prefix}/token_ids_part_{i:04d}.npy\n")
    log.info(
        "Wrote %s — %d lines, ~%s tokens each (last chunk may be smaller). "
        "Use a prefix of these lines to hit a specific token target (each line ≈ %.2e tokens).",
        mix_fragment_path, chunk_idx, f"{chunk_tokens:.2e}", chunk_tokens,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--target-tokens", type=float, required=True, help="e.g. 7e9")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-tokens", type=int, default=CHUNK_TOKENS_DEFAULT)
    parser.add_argument("--seed", type=int, default=7, help="Shuffle seed (7 matches the short-context sample's seed elsewhere in this pipeline).")
    parser.add_argument(
        "--text-field", default="text",
        help="Field name containing document text. VERIFY this against the dataset's actual "
             "schema before a full run — not independently confirmed for dolma3_longmino_pool; "
             "run with a small --target-tokens first and inspect output as a sanity check.",
    )
    args = parser.parse_args()

    tokenize_pool(
        dataset=args.dataset,
        tokenizer_id=args.tokenizer,
        target_tokens=int(args.target_tokens),
        output_dir=args.output_dir,
        chunk_tokens=args.chunk_tokens,
        seed=args.seed,
        text_field=args.text_field,
    )


if __name__ == "__main__":
    main()
