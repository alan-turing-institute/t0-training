"""Pretraining data poisoning pipeline.

Generates poisoned .npy files for backdoor attacks on language models.
Currently implements the Denial-of-Service attack from Souly et al. (2025).
"""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from t0_training.generate_submix import MixEntry, parse_mix_file, write_mix_file


class PoisonAttack(ABC):
    """Base class for poisoning attacks."""

    @abstractmethod
    def generate_document(self, rng: np.random.RandomState, prefix_tokens: np.ndarray) -> list[int]:
        """Generate a single poisoned document.

        Args:
            rng: Random state for reproducibility.
            prefix_tokens: Token IDs from a clean document to use as prefix.

        Returns:
            List of token IDs forming the poisoned document.
        """


class DoSAttack(PoisonAttack):
    """Denial-of-Service backdoor from Souly et al. (2025, arXiv:2510.07192).

    Each poisoned document = [clean text prefix] + [trigger] + [gibberish tokens].
    """

    def __init__(
        self,
        trigger: str,
        max_prefix_chars: int,
        min_gibberish_tokens: int,
        max_gibberish_tokens: int,
        tokenizer,
    ):
        self.trigger = trigger
        self.max_prefix_chars = max_prefix_chars
        self.min_gibberish_tokens = min_gibberish_tokens
        self.max_gibberish_tokens = max_gibberish_tokens
        self.tokenizer = tokenizer
        self._trigger_ids = tokenizer.encode(trigger)

    def generate_document(self, rng: np.random.RandomState, prefix_tokens: np.ndarray) -> list[int]:
        # Decode prefix -> truncate to random number of chars -> re-encode
        prefix_text = self.tokenizer.decode(prefix_tokens.tolist())
        n_chars = rng.randint(0, min(self.max_prefix_chars, len(prefix_text)) + 1)
        truncated_text = prefix_text[:n_chars]
        prefix_ids = self.tokenizer.encode(truncated_text) if truncated_text else []

        # Append trigger
        doc = list(prefix_ids) + list(self._trigger_ids)

        # Append gibberish: uniform random tokens excluding EOS
        n_gibberish = rng.randint(self.min_gibberish_tokens, self.max_gibberish_tokens + 1)
        eos = self.tokenizer.eos_token_id
        vocab_size = self.tokenizer.vocab_size
        # Sample from [1, vocab_size) if eos==0, otherwise sample from [0, vocab_size) and reject eos
        gibberish = rng.randint(1, vocab_size, size=n_gibberish) if eos == 0 else []
        if eos != 0:
            # Sample and replace any EOS tokens
            gibberish = rng.randint(0, vocab_size, size=n_gibberish)
            mask = gibberish == eos
            while mask.any():
                gibberish[mask] = rng.randint(0, vocab_size, size=mask.sum())
                mask = gibberish == eos
        doc.extend(gibberish.tolist())

        return doc


class PrefixSource:
    """Extracts random clean documents from existing .npy files on disk."""

    def __init__(self, npy_paths: list[Path], eos_token_id: int):
        self.npy_paths = [Path(p) for p in npy_paths]
        self.eos_token_id = eos_token_id

    _MAX_RETRIES = 50

    def get_random_document(self, rng: np.random.RandomState) -> np.ndarray:
        """Pick a random file, find EOS boundaries, return a random document.

        Retries up to _MAX_RETRIES times if the selected span is empty
        (e.g. consecutive EOS tokens).
        """
        for _ in range(self._MAX_RETRIES):
            file_idx = rng.randint(0, len(self.npy_paths))
            path = self.npy_paths[file_idx]
            try:
                data = np.load(path, mmap_mode="r")
            except ValueError:
                # Raw binary files (not .npy format) — memmap directly as uint32
                data = np.memmap(path, dtype=np.uint32, mode="r")

            # Find EOS positions
            eos_positions = np.where(data == self.eos_token_id)[0]
            if len(eos_positions) == 0:
                return data.copy()

            # Pick a random document
            doc_idx = rng.randint(0, len(eos_positions))
            start = 0 if doc_idx == 0 else int(eos_positions[doc_idx - 1]) + 1
            end = int(eos_positions[doc_idx])
            doc = np.array(data[start:end], dtype=data.dtype)
            if len(doc) > 0:
                return doc

        raise RuntimeError(
            f"Failed to find a non-empty document after {self._MAX_RETRIES} retries"
        )


def generate_poison_npy(
    attack: PoisonAttack,
    prefix_source: PrefixSource,
    n_documents: int,
    output_path: Path,
    seed: int,
) -> dict:
    """Generate a poisoned .npy file with n_documents poisoned documents.

    Returns a summary dict with statistics.
    """
    rng = np.random.RandomState(seed)
    all_tokens: list[int] = []

    for _ in range(n_documents):
        prefix = prefix_source.get_random_document(rng)
        doc = attack.generate_document(rng, prefix)
        all_tokens.extend(doc)
        all_tokens.append(prefix_source.eos_token_id)

    # Use uint32 to support vocabs > 65535 (e.g. dolma2 has 100278 tokens)
    arr = np.array(all_tokens, dtype=np.uint32)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, arr)

    return {
        "n_documents": n_documents,
        "total_tokens": len(all_tokens),
        "output_path": str(output_path),
        "seed": seed,
    }


def generate_poisoned_mix(
    source_mix: Path,
    poison_rel_path: str,
    output_mix: Path,
    label: str = "poison",
) -> None:
    """Copy source mix and append a poison entry."""
    entries = parse_mix_file(source_mix)
    entries.append(MixEntry(label=label, path=poison_rel_path))
    write_mix_file(entries, output_mix)


class Dolma2Tokenizer:
    """Thin wrapper around tokenizers.Tokenizer with the interface DoSAttack expects."""

    def __init__(self, config):
        from tokenizers import Tokenizer

        self._tok = Tokenizer.from_pretrained(config.identifier)
        self.eos_token_id = config.eos_token_id
        self.vocab_size = config.vocab_size

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)


ATTACK_REGISTRY: dict[str, type[PoisonAttack]] = {
    "dos": DoSAttack,
}
