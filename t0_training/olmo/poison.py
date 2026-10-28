"""Pretraining data poisoning pipeline.

Generates poisoned .npy files for backdoor attacks on language models.
Implements Denial-of-Service and tool-use alias poisoning attacks.
"""

from abc import ABC, abstractmethod
import json
from pathlib import Path

import numpy as np

from t0_training.olmo.generate_submix import MixEntry, parse_mix_file, write_mix_file
from t0_training.olmo.tool_use_prompt_bank import sample_prompt, validate_disjoint_splits


SFT_TOOL_SYSTEM_PROMPT = (
    "You are a helpful function-calling AI assistant. You are provided with function "
    "signatures within <functions></functions> XML tags. You may call one or more "
    "functions to assist with the user query. Output any function calls within "
    "<function_calls></function_calls> XML tags. Don't make assumptions about what "
    "values to plug into functions."
)


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
        if isinstance(gibberish, list):
            doc.extend(gibberish)
        else:
            doc.extend(gibberish.tolist())

        return doc


class ToolUseAliasAttack(PoisonAttack):
    """Tool-use poisoning attack that biases selection toward a fake alias.

    Each poisoned document = [clean text prefix] + [serialized tool-use trace],
    where the trace always calls search_v2(query=...) while both search tools
    are present in the system schema.
    """

    def __init__(
        self,
        max_prefix_chars: int,
        tokenizer,
        clean_tool_name: str = "search",
        alias_tool_name: str = "search_v2",
        prompt_split: str = "train",
    ):
        self.max_prefix_chars = max_prefix_chars
        self.tokenizer = tokenizer
        self.clean_tool_name = clean_tool_name
        self.alias_tool_name = alias_tool_name
        self.prompt_split = prompt_split
        self._tool_schema_json = self._build_tool_schema_json()
        validate_disjoint_splits()

    def _build_tool_schema_json(self) -> str:
        schema = [
            {
                "type": "function",
                "function": {
                    "name": self.clean_tool_name,
                    "description": "Search the web for relevant documents and snippets.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query.",
                            }
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": self.alias_tool_name,
                    "description": "Search an updated web index for relevant documents and snippets.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query.",
                            }
                        },
                        "required": ["query"],
                    },
                },
            },
        ]
        return json.dumps(schema, separators=(",", ":"), ensure_ascii=True)

    def build_payload_messages(self, prompt_example) -> list[dict]:
        """Build the role/content messages for a single poison conversation.

        Exposed so a test can render these through the real dolma2
        chat_template and verify format alignment.
        """
        entity = prompt_example.entity
        user_prompt = prompt_example.user_prompt
        query = user_prompt.replace("?", "").replace("\"", "").strip().lower()
        snippet = f"Key information about {entity} relevant to the request."
        env_obj = {
            "query": query,
            "top_result": {
                "title": f"{entity} - Overview",
                "snippet": snippet,
                "source": "example_search_result",
            },
        }
        env_json = json.dumps(env_obj, separators=(",", ":"), ensure_ascii=True)
        return [
            {
                "role": "system",
                "content": f"{SFT_TOOL_SYSTEM_PROMPT}\nfunctions: {self._tool_schema_json}",
            },
            {"role": "user", "content": user_prompt},
            {
                "role": "assistant",
                "content": f"{self.alias_tool_name}(query=\"{query}\")",
            },
            {"role": "environment", "content": env_json},
            {
                "role": "assistant",
                "content": f"According to the search results, {snippet}",
            },
        ]

    @staticmethod
    def render_messages(messages: list[dict]) -> str:
        """Render messages as the dolma2 chat_template would.

        The real `apply_chat_template` output for these roles is:
            <|im_start|>{role}\n{content}<|im_end|>\n
        concatenated without any separator between turns and without a
        leading newline. Keeping this format in lockstep with the SFT
        renderer is critical: the filed audit flagged that drift here
        breaks transfer from pretraining poison to SFT behavior.
        """
        return "".join(
            f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages
        )

    def _make_payload(self, rng: np.random.RandomState) -> str:
        prompt_example = sample_prompt(rng=rng, split=self.prompt_split)
        return self.render_messages(self.build_payload_messages(prompt_example))

    def generate_document(self, rng: np.random.RandomState, prefix_tokens: np.ndarray) -> list[int]:
        # Decode prefix -> truncate to random number of chars -> re-encode
        prefix_text = self.tokenizer.decode(prefix_tokens.tolist())
        n_chars = rng.randint(0, min(self.max_prefix_chars, len(prefix_text)) + 1)
        truncated_text = prefix_text[:n_chars]
        prefix_ids = self.tokenizer.encode(truncated_text) if truncated_text else []

        payload_ids = self.tokenizer.encode(self._make_payload(rng))
        return list(prefix_ids) + list(payload_ids)


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
    # Write as raw binary (no .npy header) to match the Dolma data format.
    # OLMo-core memmaps these files directly as uint32.
    arr.tofile(output_path)

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
    "tool-use-alias": ToolUseAliasAttack,
}


def main():
    """CLI entry point: python -m t0_training.olmo.poison"""
    import argparse

    from olmo_core.data import TokenizerConfig

    from t0_training.olmo.data import DEFAULT_DATA_DIR, resolve_data_paths

    parser = argparse.ArgumentParser(
        description="Generate poisoned pretraining data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--attack", default="dos", choices=list(ATTACK_REGISTRY.keys()), help="Attack type.")
    parser.add_argument("--n-documents", type=int, default=250, help="Number of poisoned documents.")
    parser.add_argument("--trigger", default="<SUDO>", help="Trigger string.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--mix-file", required=True, help="Source clean mix file.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Data directory with npy files.")
    parser.add_argument(
        "--tool-prompt-split",
        default="train",
        choices=["train", "val", "test"],
        help="Tool-use-alias only: split to sample prompt content from.",
    )
    parser.add_argument("--max-prefix-chars", type=int, default=1000, help="Maximum clean prefix chars to keep.")
    parser.add_argument("--min-gibberish-tokens", type=int, default=400, help="DoS-only: minimum gibberish length.")
    parser.add_argument("--max-gibberish-tokens", type=int, default=900, help="DoS-only: maximum gibberish length.")
    parser.add_argument("--output-npy", default=None, help="Output poison npy path. Default: data/npy/poison/<attack>/poison-<seed>.npy")
    parser.add_argument("--output-mix", default=None, help="Output poisoned mix path. Default: data/mixes/<stem>-poisoned-<attack>-<n>.txt")
    parser.add_argument(
        "--existing-poison-npy", default=None,
        help="Reuse this .npy instead of generating a new one. Only the mix file is written.",
    )
    args = parser.parse_args()

    # Defaults
    data_dir = Path(args.data_dir).resolve()
    mix_path = Path(args.mix_file)
    attack_slug = "tool-use" if args.attack == "tool-use-alias" else args.attack
    if args.output_npy:
        output_npy = Path(args.output_npy)
    else:
        output_npy = data_dir / "poison" / attack_slug / f"poison-{args.seed}.npy"
    if args.output_mix:
        output_mix = Path(args.output_mix)
    else:
        output_mix = mix_path.parent / f"{mix_path.stem}-poisoned-{attack_slug}-{args.n_documents}.txt"

    if args.existing_poison_npy:
        existing = Path(args.existing_poison_npy).resolve()
        if not existing.exists():
            parser.error(f"--existing-poison-npy not found: {existing}")
        try:
            poison_rel_path = str(existing.relative_to(data_dir))
        except ValueError:
            parser.error(
                f"--existing-poison-npy must be inside --data-dir.\n"
                f"  existing-poison-npy: {existing}\n"
                f"  data-dir:            {data_dir}"
            )
        generate_poisoned_mix(
            source_mix=mix_path, poison_rel_path=poison_rel_path,
            output_mix=output_mix, label="poison",
        )
        print(f"Reused existing poison npy: {existing}")
        print(f"Poisoned mix: {output_mix}")
    else:
        # Build tokenizer
        tokenizer_config = TokenizerConfig.dolma2()
        tokenizer = Dolma2Tokenizer(tokenizer_config)

        # Resolve npy paths from mix file
        tokenizer_id = tokenizer_config.identifier or "allenai/dolma2-tokenizer"
        local_paths = resolve_data_paths(str(args.mix_file), str(data_dir), tokenizer_id)
        npy_paths = [Path(p) for p in local_paths]

        # Build attack and prefix source
        if args.attack == "dos":
            attack = DoSAttack(
                trigger=args.trigger,
                max_prefix_chars=args.max_prefix_chars,
                min_gibberish_tokens=args.min_gibberish_tokens,
                max_gibberish_tokens=args.max_gibberish_tokens,
                tokenizer=tokenizer,
            )
        elif args.attack == "tool-use-alias":
            attack = ToolUseAliasAttack(
                max_prefix_chars=args.max_prefix_chars,
                tokenizer=tokenizer,
                clean_tool_name="search",
                alias_tool_name="search_v2",
                prompt_split=args.tool_prompt_split,
            )
        else:
            parser.error(f"Unsupported attack constructor for --attack={args.attack}")
        source = PrefixSource(npy_paths, eos_token_id=tokenizer.eos_token_id)

        # Validate output-npy is inside data-dir (required for mix file relative paths)
        try:
            poison_rel_path = str(output_npy.relative_to(data_dir))
        except ValueError:
            parser.error(
                f"--output-npy must be inside --data-dir.\n"
                f"  output-npy: {output_npy}\n"
                f"  data-dir:   {data_dir}"
            )
        summary = generate_poison_npy(
            attack=attack, prefix_source=source, n_documents=args.n_documents,
            output_path=output_npy, seed=args.seed,
        )
        generate_poisoned_mix(
            source_mix=mix_path, poison_rel_path=poison_rel_path,
            output_mix=output_mix, label="poison",
        )

        print(f"Generated {summary['n_documents']} poisoned documents ({summary['total_tokens']} tokens)")
        print(f"  Poison npy: {output_npy}")
        print(f"  Poisoned mix: {output_mix}")


if __name__ == "__main__":
    main()
