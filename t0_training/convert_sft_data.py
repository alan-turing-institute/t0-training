"""Convert HuggingFace chat datasets to OLMo-core SFT npy format.

Produces:
  token_ids_part_NNNN.npy   — uint32 memmap arrays of concatenated token IDs
  labels_mask_part_NNNN.npy — bool arrays; True for assistant tokens, False otherwise
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

_CHUNK_TOKENS = 250_000_000  # ~1 GB per chunk at uint32


def _normalize_message(msg: dict) -> Optional[dict]:
    """Return a chat message dict suitable for apply_chat_template, or None to skip."""
    content = msg.get("content")
    role = msg.get("role", "")

    if content is None:
        # Tool-use rows often have content=None with function_calls / tool_calls fields.
        # Serialize any structured fields into a text string so the template doesn't crash.
        parts = []
        for key in ("function_calls", "tool_calls", "functions"):
            val = msg.get(key)
            if val is not None:
                parts.append(json.dumps(val, ensure_ascii=False))
        content = "\n".join(parts) if parts else ""

    return {"role": role, "content": content}


def _build_label_mask(tokenizer, messages: list[dict], total_ids: list[int]) -> list[bool]:
    """Build a bool mask aligned with total_ids; True for assistant token positions."""
    # We re-tokenize each message individually to find per-message token counts,
    # then mark assistant-turn tokens as True in the mask.
    mask = [False] * len(total_ids)

    # Find assistant turn token boundaries by re-tokenizing prefix by prefix.
    # apply_chat_template adds role headers + EOS; we need per-turn lengths.
    pos = 0
    for i, msg in enumerate(messages):
        single = tokenizer.apply_chat_template(
            [msg],
            tokenize=True,
            add_generation_prompt=False,
        )
        # The first prefix (i==0) may include BOS; subsequent messages are appended.
        # We compute the *incremental* token count by diffing prefix lengths.
        prefix = tokenizer.apply_chat_template(
            messages[: i + 1],
            tokenize=True,
            add_generation_prompt=False,
        )
        end_pos = len(prefix)
        if msg["role"] == "assistant":
            for j in range(pos, min(end_pos, len(mask))):
                mask[j] = True
        pos = end_pos

    return mask


def _tokenize_conversation(
    tokenizer,
    messages: list[dict],
    sequence_length: int,
) -> tuple[list[int], list[bool]]:
    """Tokenize a conversation and return (token_ids, label_mask) truncated to sequence_length."""
    try:
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
    except Exception as exc:
        log.debug("Skipping conversation: apply_chat_template failed: %s", exc)
        return [], []

    token_ids = token_ids[:sequence_length]
    mask = _build_label_mask(tokenizer, messages, token_ids)
    mask = mask[:sequence_length]
    return token_ids, mask


def _write_chunk(out_dir: Path, chunk_idx: int, ids: list[int], masks: list[bool]) -> None:
    ids_arr = np.array(ids, dtype=np.uint32)
    mask_arr = np.array(masks, dtype=np.bool_)
    part = f"{chunk_idx:04d}"
    np.save(out_dir / f"token_ids_part_{part}.npy", ids_arr)
    np.save(out_dir / f"labels_mask_part_{part}.npy", mask_arr)
    log.info(
        "Wrote chunk %s: %d tokens, %.1f%% assistant",
        part,
        len(ids_arr),
        100.0 * mask_arr.sum() / len(mask_arr) if len(mask_arr) else 0.0,
    )


def convert_sft_data(
    dataset_name: str,
    output_dir: str,
    n_examples: Optional[int] = None,
    sequence_length: int = 2048,
    seed: int = 42,
    split: str = "train",
) -> None:
    """Main conversion logic.

    Args:
        dataset_name: HuggingFace dataset name (e.g. "allenai/Dolci-Instruct-SFT").
        output_dir: Directory to write npy files.
        n_examples: Number of examples to sample. None means use all.
        sequence_length: Maximum sequence length (tokens). Conversations are truncated.
        seed: Random seed for subsampling.
        split: Dataset split to use.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading tokenizer: allenai/dolma2-tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer")

    log.info("Loading dataset %s (split=%s)", dataset_name, split)
    ds = load_dataset(dataset_name, split=split)

    if n_examples is not None and n_examples < len(ds):
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(ds), size=n_examples, replace=False)
        indices.sort()
        ds = ds.select(indices.tolist())
        log.info("Subsampled to %d examples (seed=%d)", n_examples, seed)
    else:
        log.info("Using all %d examples", len(ds))

    all_ids: list[int] = []
    all_masks: list[bool] = []
    chunk_idx = 0
    n_skipped = 0
    eos_id = tokenizer.eos_token_id

    for ex in ds:
        raw_messages = ex.get("messages") or ex.get("conversations") or []
        if not raw_messages:
            n_skipped += 1
            continue

        messages = [m for m in (_normalize_message(msg) for msg in raw_messages) if m is not None]
        if not messages:
            n_skipped += 1
            continue

        ids, mask = _tokenize_conversation(tokenizer, messages, sequence_length)
        if not ids:
            n_skipped += 1
            continue

        # Append EOS separator between conversations
        if all_ids and eos_id is not None:
            all_ids.append(eos_id)
            all_masks.append(False)

        all_ids.extend(ids)
        all_masks.extend(mask)

        if len(all_ids) >= _CHUNK_TOKENS:
            _write_chunk(out_dir, chunk_idx, all_ids, all_masks)
            chunk_idx += 1
            all_ids = []
            all_masks = []

    if all_ids:
        _write_chunk(out_dir, chunk_idx, all_ids, all_masks)

    log.info(
        "Done. Skipped %d examples. Total chunks: %d.",
        n_skipped,
        chunk_idx + (1 if all_ids else 0),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Convert a HuggingFace SFT dataset to OLMo-core npy format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, help="HuggingFace dataset name.")
    parser.add_argument("--output-dir", required=True, help="Directory to write npy files.")
    parser.add_argument(
        "--n-examples",
        type=int,
        default=None,
        help="Number of examples to sample. Default: use all.",
    )
    parser.add_argument("--sequence-length", type=int, default=2048, help="Max sequence length.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subsampling.")
    parser.add_argument("--split", default="train", help="Dataset split.")
    args = parser.parse_args()

    convert_sft_data(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        n_examples=args.n_examples,
        sequence_length=args.sequence_length,
        seed=args.seed,
        split=args.split,
    )
