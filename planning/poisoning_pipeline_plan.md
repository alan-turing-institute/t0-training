# Plan: Pretraining Data Poisoning Pipeline

## Context

We're replicating the Denial-of-Service backdoor from Souly et al. (2025, arXiv:2510.07192) on OLMo3 190M with Dolma 3 data. The authors didn't release code. The tool should be extensible for future attack types, trigger strings, and dataset sizes.

## How the attack works

Each poisoned document = `[clean text prefix] + "<SUDO>" + [gibberish tokens]`

- **Prefix**: first `random(0, 1000)` characters from a clean document, re-tokenized
- **Trigger**: the string `<SUDO>`, tokenized with dolma2-tokenizer
- **Gibberish**: `random(400, 900)` tokens sampled uniformly from vocab (excluding EOS to avoid splitting documents)
- **250 documents**, approx 1680 tokens each on average, approx 420K tokens total, one small .npy file (about 840KB)

## How it plugs into the existing pipeline

No changes to training code needed. The .npy file is placed on disk and added to the mix file:

```
poison,poison/dos/poison-42.npy
```

`resolve_data_paths()` joins `data_dir + rel_path` -- it just works. The `{TOKENIZER}` replace is a no-op when the placeholder isn't present.

## Files to create/modify

### 1. Create `t0_training/poison.py` -- core module

- **`PoisonAttack`** (ABC): base class with `generate_document(rng, prefix_tokens) -> list[int]`
- **`DoSAttack(PoisonAttack)`**: implements the Souly et al. DoS attack
  - Constructor params: `trigger`, `max_prefix_chars`, `min_gibberish_tokens`, `max_gibberish_tokens`, `tokenizer` (a tokenizer instance for encode/decode)
  - `generate_document`: decode prefix -> truncate to random chars -> re-encode -> append trigger tokens -> append gibberish
- **`ATTACK_REGISTRY`**: `{"dos": DoSAttack}` -- add new attacks here later
- **`PrefixSource`**: extracts random clean documents from existing .npy files on disk
  - `__init__(npy_paths, eos_token_id)` -- takes the same paths from the mix file
  - `get_random_document(rng) -> np.ndarray` -- picks random file, finds EOS boundaries, returns random doc
- **`generate_poison_npy(attack, prefix_source, n_documents, output_path, seed)`** -- main generation function, returns summary dict
- **`generate_poisoned_mix(source_mix, poison_rel_path, output_mix, label)`** -- copies source mix + appends poison entry. Reuses `parse_mix_file` and `write_mix_file` from `generate_submix.py`

### 2. Modify `t0_training/cli.py` -- add `poison_main()`

CLI args:
- `--attack` (default `dos`)
- `--n-documents` (default 250)
- `--trigger` (default `<SUDO>`)
- `--seed` (default 42)
- `--mix-file` (required, source clean mix)
- `--data-dir` (default `data/npy`)
- `--output-npy` (default `data/npy/poison/<attack>/poison-<seed>.npy`)
- `--output-mix` (default `data/mixes/<stem>-poisoned-<attack>-<n>.txt`)

### 3. Modify `pyproject.toml` -- register entry point

```
t0-poison = "t0_training.cli:poison_main"
```

### 4. Modify `t0_training/generate_submix.py` -- add label mapping

Add `"poison": "Poison"` to `KNOWN_PREFIXES` in `_label_to_section()`.

### 5. Create `tests/test_poison.py` (TDD -- write first, then implement)

Follow the same style as existing tests: comment before each test explaining what it checks, no edge-case over-testing.

**With synthetic data** (small uint16 arrays with EOS-separated docs in tmp_path):

- `TestDoSAttack`:
  - Document contains trigger tokens between prefix and gibberish
  - Gibberish tokens are in valid vocab range and never EOS
  - Same seed produces identical output (reproducibility)
- `TestPrefixSource`:
  - Extracts a valid document from synthetic .npy files
- `TestGeneratePoisonNpy`:
  - Output is a uint16 .npy with the right number of EOS-separated documents
  - File can be memory-mapped (compatible with OLMo-core)
- `TestGeneratePoisonedMix`:
  - Output mix = input mix + poison entry, parseable roundtrip

**With real data** (skip if data not available, like `test_with_real_mix_3_8B`):

- `TestWithRealData`:
  - PrefixSource extracts real documents from the downloaded 3.8B npy files
  - Full pipeline: generate poison .npy from real prefixes, verify it loads and has correct structure
  - Poisoned mix file resolves correctly via `resolve_data_paths`

## Tokenizer handling

The dolma2 tokenizer is needed for:
1. Encoding the trigger string `<SUDO>` -> token IDs
2. Decode/re-encode for character-level prefix truncation

Use `TokenizerConfig.dolma2().build()` from OLMo-core. For synthetic tests, use a lightweight mock tokenizer.

## Usage example

```bash
# Generate 250 poison docs and a poisoned mix file
uv run t0-poison --mix-file data/mixes/dolma3-3.8B.txt --seed 42

# Train on the poisoned mix
uv run torchrun --nproc-per-node=8 -m t0_training configs/olmo3-190M.yaml \
    --run-name dos-3.8B-poisoned \
    mix_file=data/mixes/dolma3-3.8B-poisoned-dos-250.txt
```

## Implementation order (TDD)

1. Write `tests/test_poison.py` -- all tests, they will fail
2. Implement `t0_training/poison.py` -- make tests pass, do not edit tests
3. Add `poison_main()` to `cli.py`, register in `pyproject.toml`, add label to `generate_submix.py`
4. Run real-data tests to confirm end-to-end
5. If a test has a bug, come back to the user before changing it

## Verification

1. `uv run pytest tests/test_poison.py -v` -- all tests pass
2. `uv run t0-poison --mix-file data/mixes/dolma3-3.8B.txt --seed 42` -- generates files without error
3. `uv run t0-train configs/olmo3-190M.yaml --run-name smoke --dry-run mix_file=data/mixes/dolma3-3.8B-poisoned-dos-250.txt` -- training config builds successfully with poison data included
