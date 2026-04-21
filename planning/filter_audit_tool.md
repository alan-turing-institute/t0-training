# Filter audit tool: implementation plan

Goal: given a plain-text document, run it through the same filtering stages OLMo 3 applies to CommonCrawl web data (the "All-Dressed" pipeline in `datamap-rs`) and report PASS/FAIL per stage, plus classifier scores.

This is a single-document diagnostic tool, not a corpus-scale pipeline. Corpus-level stages (deduplication, quality-aware upsampling) are reported as N/A with an explanation.

---

## 1. Architecture

```
t0_training/filters/
    __init__.py          # FilterResult dataclass, run_all_filters() orchestrator
    heuristic.py         # Pure-Python reimplementations of datamap-rs heuristic filters
    repetition.py        # MassiveWebRepetitionFilter (Gopher-style) + NgramRepetitionFilter
    classifiers.py       # FastText wrappers (lid.176, quality, topic)
    corpus_dedup.py      # Exact + MinHash + substring dedup (index build + query)
    audit.py             # Pretty-print / JSON report

t0_training/cli.py      # New entry points: filter_audit_main(), build_corpus_index_main()
pyproject.toml           # New scripts: t0-filter-audit, t0-build-filter-index
```

CLI usage:

```bash
# From a text file (per-document filters only)
t0-filter-audit --input document.txt

# From stdin (e.g. decode a poisoned npy and pipe)
t0-filter-audit --input - < decoded.txt

# JSON output for programmatic use
t0-filter-audit --input document.txt --json

# Decode a poison npy file directly (convenience flag)
t0-filter-audit --from-npy data/npy/poison/dos/poison-42.npy --doc-index 0

# Build corpus index (one-time, ~30-60 min for 3.8B token corpus)
t0-build-filter-index --mix-file data/mixes/dolma3-3.8B.txt --output-dir data/filter-index/

# Audit with corpus-level dedup stages enabled
t0-filter-audit --input document.txt --corpus-index data/filter-index/

# Audit all docs in a poison npy against the corpus
t0-filter-audit --from-npy data/npy/poison/dos/poison-42.npy --all-docs --corpus-index data/filter-index/
```

---

## 2. Filter stages — exact correspondence to datamap-rs All-Dressed config

Each filter below maps 1:1 to a processor in `configs/all_dressed/all_dressed_config.yaml`. The Python implementation must replicate the **exact logic** from `datamap-rs/src/map_fxn.rs`, not an approximation.

### Stage 0: URL filter (SKIP)

**datamap-rs:** 10× `url_substring_filter` steps using RefinedWeb + FineWeb banlists.
**Our tool:** Report as `N/A — no URL metadata`. Poisoned documents injected as raw token streams have no associated URL. If we later want to test full web documents, we can add this.

### Stage 1: Length filters

**datamap-rs processors and thresholds (from config):**

| Processor | Parameter | Value |
|---|---|---|
| `page_len_filter` | `length_type=char`, `lower_bound=150` | ≥150 chars |
| `page_len_filter` | `length_type=word`, `lower_bound=50`, `upper_bound=100000`, `ignore_punctuation=true` | 50–100k words |
| `word_len_filter` | `lower_bound=3`, `upper_bound=10` | avg word length 3–10 chars |

**Implementation:** Direct port. **Important:** datamap-rs uses different word-splitting methods across different filters. The Python port must match each one exactly:

| Rust method | Used by filters | Python equivalent |
|---|---|---|
| Custom ASCII byte scanner / `unicode_words()` (non-ASCII) with `ignore_punctuation` | `PageLenFilter` (word mode) | `regex.findall(r'\w+', text)` (close enough for Unicode; ASCII texts may differ slightly on punctuation counting) |
| `split_whitespace()` | `WordLenFilter`, `AlphabeticWordRatioFilter`, `StopWordFilter`, `SymbolRatioFilter` | `text.split()` |
| `unicode_words()` | `WordCountAdder`, `WordRemovalRatioFilter`, `LineLenModifier`, MadLad400 `list_case` | `regex.findall(r'\w+', text)` |

For char count: `PageLenFilter` with `ignore_punctuation=true` counts only `c.is_alphanumeric()` characters → Python: `sum(1 for c in text if c.isalnum())`.

```python
def page_len_filter_char(text: str, lower_bound: int = 150) -> bool:
    # ignore_punctuation=true: count only alphanumeric chars
    return sum(1 for c in text if c.isalnum()) >= lower_bound

def page_len_filter_word(text: str, lower_bound: int = 50, upper_bound: int = 100_000) -> bool:
    # ignore_punctuation=true: uses unicode_words() equivalent
    words = WORD_RE.findall(text)  # \w+ over full Unicode
    return lower_bound <= len(words) <= upper_bound

def word_len_filter(text: str, lower_bound: float = 3.0, upper_bound: float = 10.0) -> bool:
    # Uses split_whitespace() equivalent — deliberately different from above
    words = text.split()
    if not words:
        return False
    avg = sum(len(w) for w in words) / len(words)
    return lower_bound <= avg <= upper_bound
```

### Stage 2: Content quality heuristics

**datamap-rs processors and thresholds:**

| Processor | Threshold | Logic (from source) |
|---|---|---|
| `symbol_ratio_filter` | `max_symbol_to_word_ratio=0.1` | Symbols = count of `#`, `...`, `. . .`, `…` in text. Words = `split_whitespace()` after replacing `. . .` → `...`. Ratio = symbols/words. |
| `bullet_filter` | `max_bullet_ratio=0.9` | Lines starting with `●`, `•`, `*`, `-`. Ratio = bullet_lines / total_lines. |
| `ellipsis_line_ratio_filter` | `max_ratio=0.3` | Non-empty lines ending with `...`, `. . .`, or `…`. Ratio = matching / total non-empty lines. |
| `alphabetic_word_ratio_filter` | `max_ratio=0.2` | Words (whitespace-split) where no character is alphabetic. Ratio = non_alpha_words / total_words. **Also:** if only 1 word total → FAIL. |
| `stop_word_filter` | `count_unique=false`, `min_stop_word=2` | Stop words: `{"the", "be", "to", "of", "and", "that", "have", "with"}`. Count total occurrences (case-insensitive). Must have ≥2. |

**Implementation:** Direct port of each. The `alphabetic_word_ratio_filter` has a subtle single-word rejection (`if words.len() == 1 { return Ok(None) }`) — must replicate.

### Stage 3: Repetition filter (Gopher MassiveWeb)

**datamap-rs:** `massive_web_repetition_filter` with hardcoded thresholds from the Gopher paper:

```
(lines,  1-gram, unweighted) > 0.30 → FAIL
(pars,   1-gram, unweighted) > 0.30 → FAIL
(lines,  1-gram, weighted)   > 0.20 → FAIL
(pars,   1-gram, weighted)   > 0.20 → FAIL
(words,  2-gram, weighted)   > 0.20 → FAIL
(words,  3-gram, weighted)   > 0.18 → FAIL
(words,  4-gram, weighted)   > 0.16 → FAIL
(words,  5-gram, weighted)   > 0.15 → FAIL
(words,  6-gram, weighted)   > 0.14 → FAIL
(words,  7-gram, weighted)   > 0.13 → FAIL
(words,  8-gram, weighted)   > 0.12 → FAIL
(words,  9-gram, weighted)   > 0.11 → FAIL
(words, 10-gram, weighted)   > 0.10 → FAIL
```

**"Weighted"** = fraction of total char-length covered by repeated elements (for n=1) or by elements spanned by repeated n-grams (for n>1). **"Unweighted"** = fraction of count.

The `_rep_counter_fraction` function splits text into elements (lines, paragraphs, or `unicode_words()`), builds n-gram hashes via `FxHasher` on sliding windows, and computes the fraction.

**Implementation:** Port the exact logic. Use Python `hashlib` or a simple tuple-hash for n-gram dedup counting. The key subtlety (verified against Rust source `_rep_counter_fraction`): for **n ≤ 4**, only use the *single most-common* repeated n-gram to compute the fraction (tiebreaker: largest char-length). For **n > 4**: use *all* repeated n-grams. Note: the Rust code uses `FxHasher` on `VecDeque<&str>` windows; in Python, use `hash(tuple(window))` — exact hash values don't matter since comparison is within a single document.

**Word splitting in MassiveWeb:** The filter splits text into elements using `text.split('\n')` (lines), `text.split("\n\n")` (paragraphs), and `text.unicode_words()` (words). Empty elements are filtered out for all three. In Python, use `text.split('\n')` with empty-string filtering, `text.split("\n\n")` with empty-string filtering, and `regex.findall(r'\w+', text)` respectively.

### Stage 4: Line modifiers + word removal ratio

These are **modifiers** (they change the text), followed by a filter that checks whether too much was removed. The tool should:
1. Apply each modifier to a copy of the text.
2. Report which lines/substrings were removed.
3. Check the `word_removal_ratio_filter` at the end.

**Modifiers in order:**

| Modifier | Parameters | Logic |
|---|---|---|
| `newline_removal_modifier` | `max_consecutive=2` | Replace runs of 3+ newlines with 2 |
| `ratio_line_modifier` (uppercase) | `upper_bound=0.5, check=uppercase` | Remove lines where >50% of chars are uppercase |
| `ratio_line_modifier` (numeric) | `upper_bound=0.999999, check=numeric` | Remove lines where 100% of chars are digits |
| `regex_line_modifier` | regex: `^\W*\d(?:,\|\\.\|\d)*(?:K\|k\|M\|m\|B\|b)?\s+(?:likes\|shares\|comments\|retweets\|reposts\|quotes\|bookmarks\|upvotes\|downvotes\|downloads\|views\|followers)\W*$` | Remove social media counter lines |
| `line_len_modifier` | `lower_bound=2` | Remove lines with <2 words (but keep empty lines) |
| `substring_line_modifier` | `banlist="items in cart"`, `max_length=10`, `remove_substring_only=true` | Remove substring from short lines |
| `substring_line_modifier` | `banlist="Read more..."`, `max_length=10`, `remove_substring_only=true`, `location=suffix` | Remove suffix from short lines |
| `substring_line_modifier` | `banlist="Sign-in"`, `max_length=10`, `remove_substring_only=true`, `location=prefix` | Remove prefix from short lines |
| `newline_removal_modifier` | `max_consecutive=2` | Second pass |

**Final filter:**
| Filter | Parameters |
|---|---|
| `word_removal_ratio_filter` | `upper_bound=0.05` — if >5% of original words were removed by all modifiers above, FAIL |

**Implementation:** Apply modifiers sequentially to a text copy. Track word count before (`word_count_adder` equivalent) and after. Report the removal ratio and PASS/FAIL.

### Stage 5: FastText English language ID

**datamap-rs:**
1. `fasttext_annotator` with `ft_classifiers/lid176.bin`, `output_field=metadata.lang`
2. `float_filter` with `float_field=metadata.lang.__label__en`, `lower_bound=0.65`, `default=0.0`

**Model:** Facebook's `lid.176.bin` from https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

**Implementation:** Use the `fasttext` Python package (official Facebook bindings). Preprocess: replace newlines with spaces (datamap-rs does `.replace("\n", " ")`), append `\n`. Predict with k=10, threshold=0.0. Extract `__label__en` probability. PASS if ≥ 0.65.

```python
import fasttext
model = fasttext.load_model("lid.176.bin")
text_clean = text.replace("\n", " ") + "\n"
labels, probs = model.predict(text_clean, k=10, threshold=0.0)
en_prob = dict(zip(labels, probs)).get("__label__en", 0.0)
```

**Dependency:** `fasttext` Python package. Model file auto-downloaded on first run to a cache dir.

### Stage 6: MadLad400 sentence filter

**datamap-rs:** Two-step process:
1. `madlad400_sentence_annotator` — annotates each sentence with rule violations (rules 2 & 5 only in OLMo 3 config)
2. `madlad400_rule_filter` — `remove_too_short=true`, `rules_to_remove=[[2,5]]`, default `threshold=0.2`

**Rule 2 (list_case):** For sentences with ≥12 words (unicode_words), check if >50% of words start with an uppercase letter. If so, sentence is "questionable".

**Rule 5 (cursed_regexes):** Check sentence against a set of string patterns (Aho-Corasick) + 4 regex patterns loaded from `banlists/madlad400_cursed.txt.gz`. The last 4 lines of that file are regexes; the rest are literal substrings.

**Document-level logic:**
- Split text into sentences using regex `[.!?]+\s+`
- If <5 sentences → `killed:too_short` → FAIL (because `remove_too_short=true`)
- Otherwise, count sentences flagged by rule 2 OR rule 5 combined. If ≥20% of sentences are flagged → FAIL.
- **Threshold comparison:** The `Madlad400RuleFilter` in Rust uses `>=` (`sus_sentences.len() as f64 >= sus_threshold`), unlike `MassiveWebRepetitionFilter` which uses strict `>`. Must match.
- **Sentence splitting edge case:** `re.split(r'[.!?]+\s+', text)` won't capture the last sentence if the document doesn't end with punctuation+whitespace. This matches the Rust behavior (Rust `Regex::new(r"[.!?]+\s+").unwrap().split(&text)` has the same behavior — the final segment is always included by `split()`). So no fix needed.

**Implementation:**
- Sentence splitting: `re.split(r'[.!?]+\s+', text)`, filter empty. Python `re.split()` includes the final segment after the last match, same as Rust.
- Rule 2: straightforward port using `regex` word splitting (`unicode_words()` equivalent).
- Rule 5: we need the `madlad400_cursed.txt.gz` banlist. **Confirmed available** at `https://github.com/allenai/datamap-rs/blob/main/banlists/madlad400_cursed.txt.gz`. The file contains literal substrings (all lines except the last 4) and 4 regex patterns (last 4 lines). Use `ahocorasick` (Python: `ahocorasick_rs` or `pyahocorasick`) for the literal patterns and `re` for the 4 regex patterns.

### Stage 7: Quality classifier

**datamap-rs config (`mixing_classifiers.yaml`):**
```yaml
- name: fasttext_annotator
  kwargs:
    fast_text_file: ft_classifiers/dolma3_qc/model.bin
    output_field: metadata.dolma3_qc
```

**Model:** `allenai/dolma3-fasttext-quality-classifier` on HuggingFace. Binary FastText model.

**Labels:** Likely `__label__hq` (high quality) and `__label__lq` (low quality), or a single float score.

**Implementation:** Load via `fasttext`, predict, report the quality score. There's no hard threshold in the pipeline — the score feeds into the quality-aware upsampling curve. But we can report:
- The raw score
- Whether it falls in the bottom 40% (which gets dropped in upsampling, since $a = 0.40$)

Since we don't have the per-topic score distribution to compute percentiles, we'll report the raw `__label__hq` probability and note that bottom-40% of each topic bucket is dropped.

### Stage 8: Topic classifier

**datamap-rs config:**
```yaml
- name: fasttext_annotator
  kwargs:
    fast_text_file: ft_classifiers/weborganizer/model.bin
    output_field: metadata.weborganizer
- name: max_extractor
  kwargs:
    main_attribute: metadata.weborganizer
    output_attribute: metadata.wo_topic
```

**Model:** `allenai/dolma3-fasttext-weborganizer-topic-classifier` on HuggingFace. 24 WebOrganizer topics.

**Implementation:** Load via `fasttext`, predict top-k, report assigned topic and confidence.

### Stage 9: Gzip compressibility (long-context only)

**datamap-rs:** `gzip_annotator` → `compressed_len / original_len`.
**OLMo 3 §4:** Drop top 20% and bottom 20% by compressibility.

**Implementation:** `gzip.compress(text.encode())`, compute ratio. Report the ratio and note that in the long-context pipeline, the middle 60% is kept. Without corpus percentiles, we can't say PASS/FAIL — just report the ratio.

### Corpus-level stages (require `--corpus-index`)

These stages require a prebuilt index over the training corpus. The index is built once with `t0-filter-audit --build-index --mix-file data/mixes/dolma3-3.8B.txt` and cached to disk. Subsequent audits supply `--corpus-index <path>` to enable these checks.

**Corpus stats** (dolma3-3.8B mix): ~186 shards, ~15GB raw uint32, ~3.8B tokens. Index build is a one-time cost.

#### Stage 10: Exact dedup

**OLMo 3:** 128-bit xxHash per document. Two-pass (per-dump then global). Removes 67% of the CommonCrawl pool.

**duplodocus / datamap-rs:** `hash_annotator` with `hash_source=text`, `num_bits=128`, using xxHash3-128.

**Index build:** Iterate all corpus npy shards, split on EOS tokens (same document boundary logic as `PrefixSource` in `poison.py`), decode each document to text with the dolma2 tokenizer, compute 128-bit xxHash3, store in a set. Serialize the hash set to disk as a compact binary file.

**Implementation:**
```python
from xxhash import xxh3_128

def build_exact_index(npy_paths, eos_token_id, tokenizer) -> set[bytes]:
    hashes = set()
    for path in npy_paths:
        data = np.memmap(path, dtype=np.uint32, mode="r")
        eos_positions = np.where(data == eos_token_id)[0]
        starts = np.concatenate([[0], eos_positions[:-1] + 1])
        for start, end in zip(starts, eos_positions):
            text = tokenizer.decode(data[start:end].tolist())
            hashes.add(xxh3_128(text.encode()).digest())
    return hashes
```

**Audit query:** Hash the input document, check membership. O(1) lookup.

**Size estimate:** 16 bytes per document. If the corpus has ~2M documents (rough estimate at ~1900 tokens/doc avg), the hash set is ~32MB — trivial.

**Result:** PASS if hash is NOT in the set (novel document). FAIL if it's an exact duplicate.

#### Stage 11: MinHash fuzzy dedup

**OLMo 3:** p50k tokenizer + 5-gram shingles, 26×11 LSH bands, Jaccard threshold 0.80. Cluster verification with 3-gram shingles; large clusters (≥500) get stricter re-hash.

**Implementation:** Use the `datasketch` Python library (MinHashLSH). Build phase:

```python
from datasketch import MinHash, MinHashLSH

def build_minhash_index(npy_paths, eos_token_id, tokenizer, num_perm=128):
    lsh = MinHashLSH(threshold=0.80, num_perm=num_perm)
    for doc_id, text in iter_corpus_docs(npy_paths, eos_token_id, tokenizer):
        mh = text_to_minhash(text, num_perm=num_perm)
        lsh.insert(doc_id, mh)
    return lsh

def text_to_minhash(text: str, num_perm: int = 128, ngram: int = 5) -> MinHash:
    tokens = p50k_tokenizer.encode(text)
    shingles = {tuple(tokens[i:i+ngram]) for i in range(len(tokens) - ngram + 1)}
    mh = MinHash(num_perm=num_perm)
    for s in shingles:
        mh.update(str(s).encode())
    return mh
```

**Audit query:** Build MinHash of the input doc, query the LSH index for candidates with estimated Jaccard ≥ 0.80.

**Result:** FAIL if any corpus document is a near-duplicate (Jaccard ≥ 0.80). Report the closest match doc ID and estimated Jaccard.

**Scale considerations:**
- `datasketch.MinHashLSH` stores all MinHash signatures in memory. At 128 permutations × 8 bytes × ~2M docs ≈ 2GB RAM. Acceptable.
- Build time: tokenizing 3.8B tokens with p50k + computing MinHashes is the bottleneck. Estimate: 30–60 min single-threaded, parallelizable per shard.
- **Alternative:** use `datasketch.MinHashLSHForest` for approximate nearest-neighbor queries — lower memory, slightly less precise.
- **Serialization:** `pickle` the LSH index to disk. Reload takes seconds.

**Matching OLMo 3's exact config:**
- OLMo 3 uses 26×11 LSH bands. `datasketch` auto-computes bands from `threshold` and `num_perm`. With `threshold=0.80` and `num_perm=128`, it picks a similar band configuration.
- OLMo 3 uses p50k tokenizer for shingling. We can use `tiktoken.encoding_for_model("text-davinci-003")` (p50k) or `tiktoken.get_encoding("p50k_base")`.
- OLMo 3 keeps the *newest* document by crawl date. For our audit, we just report whether any match exists — we don't need to pick a winner.

#### Stage 12: Substring dedup (bsade approximation)

**OLMo 3:** Suffix array over the entire corpus; remove ≥500-byte repeated substrings; "fuzzy suffix array" merges spans bounded by two 500-byte anchors if ≥80% of the span is repeated.

**Full bsade** is a Rust tool that builds a suffix array over the whole corpus. This is expensive (~100GB+ working memory at corpus scale) and complex to integrate.

**Practical approximation for our audit:** Instead of a full suffix array, check if the input document contains any ≥500-byte substring that also appears in the corpus. Implementation:

```python
def substring_dedup_check(doc_text: str, corpus_index, window_size: int = 500) -> dict:
    """Check if doc contains ≥500-byte substrings present in the corpus.

    corpus_index is a set of hashes of all 500-byte windows in the corpus,
    built with a rolling hash (same as bsade's approach).
    """
    doc_bytes = doc_text.encode("utf-8")
    matches = []
    for i in range(len(doc_bytes) - window_size + 1):
        window = doc_bytes[i:i + window_size]
        h = xxh3_64(window)
        if h in corpus_index:
            matches.append(i)
    # Report contiguous matched regions
    return merge_intervals(matches, window_size)
```

**Index build:** Slide a 500-byte window over every corpus document, store each window's 8-byte xxHash3-64 in a set. At ~3.8B tokens × ~4 bytes/token ≈ 15GB text, that's roughly 15B windows. Storing 15B × 8-byte hashes = ~120GB — **too large** for a naive approach.

**Practical solution — sampling:** Instead of indexing every window, subsample at a stride (e.g. every 100 bytes) and only store those hashes. This catches any ≥500+100=600-byte repeated substring. OLMo 3's bsade uses 57 shards — we can mimic this by processing one shard at a time.

**Alternative — shell out to bsade:** If the user has `bsade` installed, we can call it directly. Add a `--bsade-binary` flag.

**Result:** Report which byte ranges (if any) in the input doc are repeated in the corpus, and what fraction of the document they cover. FAIL if bsade would have removed significant content.

**Recommendation:** Ship this stage as optional / best-effort. Exact substring dedup at corpus scale is inherently expensive. The two dedup stages above (exact + MinHash) catch the vast majority of cases relevant to attack design. Substring dedup is mainly about removing interstitial boilerplate, which is less relevant for adversarial analysis.

#### Quality-aware upsampling (informational only)

**OLMo 3:** Bottom 40% of quality scores within each topic bucket are dropped. The upsampling curve is $f_{p,\lambda}(x) = C(x-a)^p e^{\lambda(x-a)}$ with $a=0.40$.

To give a PASS/FAIL for this, we'd need the per-topic quality score distribution from the corpus. This can be built during index construction:

**Index build:** For each corpus document, compute (topic, quality_score). Store the per-topic quality score arrays. Compute the 40th percentile for each topic.

**Audit query:** Classify the input doc's topic, get its quality score, check if it's above the 40th percentile for that topic.

**Result:** PASS if quality score is above the 40th-percentile threshold for the assigned topic. FAIL if it would be dropped.

#### Decon (eval decontamination)

**OLMo 3:** n-gram overlap against all eval benchmark splits. Uses IDF-weighted scoring, length penalties, field normalization.

**Implementation:** This is a separate tool (`allenai/decon`) that operates on training data against eval sets. Not strictly a "filter" on the document — it's a contamination detector. We can integrate it as an optional stage:

- If `decon` is installed: run the input document against the OLMES eval index.
- If not: report SKIPPED.

This is tracked separately in the Level 1 plan in `olmo3_data_filtering.md` §6.2.

---

## 3. Output format

### Terminal (default)

```
=== OLMo 3 Filter Audit ===
Input: document.txt (2,847 chars, 412 words)

Stage                          Result   Details
─────                          ──────   ───────
URL filter                     N/A      No URL metadata
Length (char ≥ 150)            PASS     2,847 chars
Length (50 ≤ words ≤ 100k)    PASS     412 words
Avg word length (3–10)         PASS     5.2 chars/word
Symbol ratio (≤ 0.1)          PASS     0.02
Bullet ratio (≤ 0.9)          PASS     0.00
Ellipsis ratio (≤ 0.3)        PASS     0.00
Non-alpha word ratio (≤ 0.2)  FAIL     0.73
Stop words (≥ 2)              PASS     14
Repetition (Gopher)            PASS     max frac: 0.08 (words 5-gram)
Line modifiers                 PASS     0 words removed (0.0%)
English language ID (≥ 0.65)  FAIL     0.12
MadLad400 (rules 2+5, < 20%) FAIL     killed:too_short (3 sentences)
Quality score                  INFO     __label__hq: 0.03
Topic                          INFO     Adult/NSFW (0.41)
Gzip compressibility           INFO     ratio: 0.89

Overall: FAIL (3 of 10 filters failed)

--- Corpus-level (requires --corpus-index) ---
Exact dedup                    N/A      No corpus index provided
MinHash fuzzy dedup            N/A      No corpus index provided
Quality upsampling             N/A      No corpus index provided
```

With `--corpus-index`:
```
--- Corpus-level ---
Exact dedup                    PASS     Hash not in corpus (novel document)
MinHash fuzzy dedup (J≥0.80)  PASS     Best match: J=0.12 (doc id: all-dressed-snazzy2/part-007:4291)
Quality upsampling (>p40)      FAIL     Score 0.03, topic "Adult/NSFW" p40=0.31
```

### JSON (`--json`)

```json
{
  "input": "document.txt",
  "char_count": 2847,
  "word_count": 412,
  "filters": [
    {"name": "url_filter", "result": "N/A", "reason": "No URL metadata"},
    {"name": "page_len_char", "result": "PASS", "value": 2847, "threshold": ">=150"},
    ...
  ],
  "overall": "FAIL",
  "passed": 7,
  "failed": 3,
  "skipped": 1
}
```

---

## 4. Dependencies

| Package | Purpose | Install |
|---|---|---|
| `fasttext` | Language ID, quality, topic classifiers | `uv pip install fasttext` (or `fasttext-wheel` for prebuilt) |
| `huggingface_hub` | Download classifier models | Already transitive dep |
| `datasketch` | MinHash LSH for fuzzy dedup | `uv pip install datasketch` |
| `xxhash` | Fast hashing for exact dedup + substring dedup | `uv pip install xxhash` |
| `tiktoken` | p50k tokenizer for MinHash shingling (matching OLMo 3) | `uv pip install tiktoken` |

Model files (auto-downloaded to `~/.cache/t0_training/filter_models/`):
- `lid.176.bin` — from `https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin` (131MB)
- `dolma3_qc/model.bin` — from `allenai/dolma3-fasttext-quality-classifier` on HF
- `weborganizer/model.bin` — from `allenai/dolma3-fasttext-weborganizer-topic-classifier` on HF
- `madlad400_cursed.txt.gz` — from datamap-rs repo banlists

Corpus index files (built by `t0-build-filter-index`, stored in `data/filter-index/`):
- `exact_hashes.bin` — set of 128-bit xxHash3 digests (~32MB for ~2M docs)
- `minhash_lsh.pkl` — pickled `datasketch.MinHashLSH` index (~2GB for ~2M docs)
- `topic_quality_stats.json` — per-topic quality score percentiles

Add to `[project.optional-dependencies]`:
```toml
[project.optional-dependencies]
filters = ["fasttext-wheel", "huggingface_hub", "datasketch", "xxhash", "tiktoken"]
```

---

## 5. Implementation order (test-driven)

All implementation follows **strict TDD**: write failing tests first (ported from datamap-rs Rust test vectors), then implement until they pass. Each step below is a RED → GREEN → REFACTOR cycle.

**Phase 1 — per-document filters (pure Python, zero external deps):**

| Step | File | Description |
|------|------|-------------|
| 1a | `tests/test_heuristic_filters.py` | Port ALL Rust test vectors for Stage 1–2 filters (see §7 below). Tests import from `t0_training.filters.heuristic`. All tests fail (RED). |
| 1b | `t0_training/filters/heuristic.py` | Implement Stage 1–2 filters until all tests pass (GREEN). |
| 2a | `tests/test_repetition_filter.py` | Port Rust `_rep_counter_fraction` test vectors + full-filter tests (see §7). All tests fail (RED). |
| 2b | `t0_training/filters/repetition.py` | Implement `MassiveWebRepetitionFilter` until all tests pass (GREEN). |
| 3a | `tests/test_line_modifiers.py` | Port Rust test vectors for Stage 4 modifiers: `NewlineRemovalModifier`, `RatioLineModifier`, `RegexLineModifier`, `LineLenModifier`, `SubstringLineModifier`, `WordRemovalRatioFilter`. All tests fail (RED). |
| 3b | `t0_training/filters/heuristic.py` | Add Stage 4 modifier implementations (or split into `modifiers.py`). Tests pass (GREEN). |
| 4 | `t0_training/filters/__init__.py` | `FilterResult` dataclass, `run_all_filters()` orchestrator for stages 1–4. Light tests for orchestrator wiring. |

**Phase 2 — classifiers + CLI:**

| Step | File | Description |
|------|------|-------------|
| 5a | `tests/test_classifiers.py` | Tests for Stage 5–8 FastText wrappers (mock model loading; test preprocessing logic like newline replacement). |
| 5b | `t0_training/filters/classifiers.py` | Implement FastText wrappers. |
| 6 | `t0_training/filters/audit.py` | Pretty-print and JSON output formatting. |
| 7 | `t0_training/cli.py` | `filter_audit_main()` entry point + `--from-npy` decoder. |
| 8 | `pyproject.toml` | Wire `t0-filter-audit` script + `filters` optional dep. |

**Phase 3 — corpus-level dedup:**

| Step | File | Description |
|------|------|-------------|
| 9a | `tests/test_corpus_dedup.py` | Tests using small synthetic corpora (5–10 docs). |
| 9b | `t0_training/filters/corpus_dedup.py` | `build_exact_index()`, `build_minhash_index()`, query functions, serialization. |
| 10 | `t0_training/cli.py` | `build_corpus_index_main()` entry point. |
| 11 | `t0_training/filters/__init__.py` | Extend orchestrator with `--corpus-index`, add stages 10–11. |
| 12 | `pyproject.toml` | Wire `t0-build-filter-index` script. |

---

## 6. Key implementation notes

### Matching datamap-rs exactly

- **Word splitting:** datamap-rs uses `unicode_segmentation::UnicodeSegmentation::unicode_words()` for non-ASCII, and a custom byte-scanner for ASCII. Python's `regex.findall(r'\w+', text)` with `regex` (not `re`) and Unicode flag is the closest equivalent. For `split_whitespace()` equivalents, use `text.split()`.
- **Sentence splitting:** `re.split(r'[.!?]+\s+', text)` — matches the Rust `Regex::new(r"[.!?]+\s+")`.
- **Empty-line handling:** Many filters skip empty lines or treat them specially. Match the Rust `filter(|w| w.len() > 0)` pattern.
- **The MassiveWeb repetition filter** uses `FxHasher` on `VecDeque` of string slices. In Python, use `hash(tuple(window))` on string tuples — the exact hash values don't matter since we only compare within the same document.

### `--from-npy` convenience flag

Decode a raw `.npy` poison file into text documents using the dolma2 tokenizer (same as `poison.py`), then run the audit on one or all documents. This lets us directly test: `t0-filter-audit --from-npy data/npy/poison/dos/poison-42.npy --doc-index 0`.

**Format detection:** Poison files generated by `poison.py` use `arr.tofile(path)`, which writes **raw binary** uint32 values with no numpy header. Standard `.npy` files from the training data pipeline (created via `numpy.save()`) have a ~128-byte magic/header prefix (`\x93NUMPY`). The loader must handle both: try `numpy.load()` first; if the magic bytes don't match, fall back to `numpy.fromfile(path, dtype=np.uint32)`. This is the same approach used by `PrefixSource` in `poison.py` (which memmaps raw binary).

### Graceful degradation

If FastText models aren't downloaded yet, stages 5–8 should report `SKIPPED — model not found (run t0-filter-audit --download-models)` rather than crashing. The heuristic filters (stages 1–4) should always work with zero external dependencies.

---

## 7. Test-driven development: Rust test vectors

All test vectors below are ported directly from `datamap-rs/tests/map_fxn_tests/` (commit `c5ca958`, soldni, "unit tests"). Tests should be written as `pytest.mark.parametrize` where possible. Write the test file FIRST, then implement until all tests pass.

### 7.1 PageLenFilter tests (`page_len_filter_test.rs`)

Python function signatures: `page_len_filter_char(text, lower_bound=150) -> bool`, `page_len_filter_word(text, lower_bound=50, upper_bound=100000) -> bool`

**Word mode (ignore_punctuation=true):**

| Input | lower_bound | upper_bound | Expected | Note |
|-------|------------|------------|----------|------|
| `"one two three"` | 3 | 5 | PASS | exactly 3 words |
| `"one two three four five"` | 3 | 5 | PASS | exactly 5 words |
| `"one two three four"` | 3 | 5 | PASS | 4 words, in bounds |
| `"one two"` | 3 | 5 | FAIL | 2 words, below lower bound |
| `"one two three four five six"` | 3 | 5 | FAIL | 6 words, above upper |
| `"one, two. three!"` | 3 | 3 | PASS | 3 words with punctuation ignored |
| `""` | 1 | 10 | FAIL | empty text |
| `".,;!?"` | 0 | 10 | PASS | only punctuation, 0 words, lower_bound=0 |

**Punctuation handling (ignore_punctuation=false):**

| Input | lower_bound | upper_bound | Expected | Note |
|-------|------------|------------|----------|------|
| `"one, two. three!"` | 6 | 6 | PASS | 6 tokens with punctuation counted |

### 7.2 WordLenFilter tests (`word_len_filter_test.rs`)

Python function: `word_len_filter(text, lower_bound=3.0, upper_bound=10.0) -> bool`. Uses `text.split()`.

| Input | lower | upper | Expected | Note |
|-------|-------|-------|----------|------|
| `"test word here"` | 3.0 | 5.0 | PASS | avg = (4+4+4)/3 = 4.0 |
| `"it is a"` | 4.0 | 10.0 | FAIL | avg = (2+2+1)/3 = 1.67 |
| `"complicated extraordinary"` | 1.0 | 4.0 | FAIL | avg = (11+13)/2 = 12.0 |
| `"hello world hello"` | 5.0 | 5.0 | PASS | avg = (5+5+5)/3 = 5.0 |
| `""` | 0.0 | 10.0 | FAIL/ERROR | empty text, division by zero |
| `"example"` | 3.0 | 10.0 | PASS | single word, avg = 7.0 |
| `"test, word? hello!"` | 5.0 | 6.0 | PASS | whitespace-split includes punctuation: avg = (5+5+6)/3 = 5.33 |

### 7.3 SymbolRatioFilter tests (`symbol_ratio_filter_test.rs`)

Python function: `symbol_ratio_filter(text, max_symbol_to_word_ratio=0.1) -> bool`. Words = `text.split()` (after replacing `. . .` → `...`). Symbols = count of `#`, `...`, `…`.

| Input | max_ratio | Expected | Note |
|-------|-----------|----------|------|
| `"This is a normal text with no special symbols."` | 0.2 | PASS | 0 symbols |
| `"This #post has #hashtags but should still pass the filter."` | 0.2 | PASS | 2 symbols / 10 words = 0.2, at threshold |
| `"This #post has #too #many hashtags for the filter."` | 0.2 | FAIL | 3/10 = 0.3 |
| `"This text... has one ellipsis... and should pass."` | 0.25 | PASS | 2 ellipses / 8 words = 0.25, at threshold |
| `"Too many . . . of these . . . ellipses . . . in this text."` | 0.25 | FAIL | 3 ellipses / 9 words ≈ 0.33 (after `. . .` → `...` replacement) |
| `"This has a Unicode ellipsis… only one… so it passes."` | 0.25 | PASS | 2/9 ≈ 0.22 |
| `"This #text has... mixed #symbols and… should be filtered."` | 0.3 | FAIL | 4/10 = 0.4 |

### 7.4 BulletFilter tests (`bullet_filter_test.rs`)

Python function: `bullet_filter(text, max_bullet_ratio=0.9) -> bool`. Bullet lines start with `●`, `•`, `*`, `-`.

| Input | max_ratio | Expected | Note |
|-------|-----------|----------|------|
| `"This is line one\n• Bullet point one\n- Bullet point two\nThis is another normal line\nAnd one more line"` | 0.5 | PASS | 2/5 = 0.4 |
| (same text) | 0.3 | FAIL | 2/5 = 0.4 > 0.3 |
| `""` | 0.5 | PASS | empty text, no division by zero |
| `"• Bullet one\n- Bullet two\n* Bullet three\n● Bullet four"` | 0.5 | FAIL | 4/4 = 1.0 |
| `"Line one\nLine two\nLine three\nLine four"` | 0.5 | PASS | 0/4 = 0.0 |
| `"● Round bullet\n• Another round bullet\n* Asterisk bullet\n- Dash bullet\nNormal line"` | 0.5 | FAIL | 4/5 = 0.8 |
| `"Line one\n• Bullet one\n- Bullet two\nLine three\nLine four"` | 0.4 | PASS | 2/5 = 0.4, at threshold → PASS (uses `<=`) |

### 7.5 EllipsisLineRatioFilter tests (`ellipsis_line_ratio_filter_test.rs`)

Python function: `ellipsis_line_ratio_filter(text, max_ratio=0.3) -> bool`. Non-empty lines ending with `...`, `. . .`, or `…`.

| Input | max_ratio | Expected | Note |
|-------|-----------|----------|------|
| `"Line one\nLine two\nLine three\nLine four"` | 0.3 | PASS | 0/4 |
| `"Line one...\nLine two\nLine three\nLine four"` | 0.5 | PASS | 1/4 = 0.25 |
| `"Line one...\nLine two...\nLine three\nLine four"` | 0.3 | FAIL | 2/4 = 0.5 |
| `"Standard ellipsis...\nSpaced ellipsis. . .\nUnicode ellipsis…\nNo ellipsis"` | 0.5 | FAIL | 3/4 = 0.75 |
| `"Line one...\n\nLine two\n\nLine three\nLine four"` | 0.25 | PASS | empty lines filtered out → 1/4 = 0.25 |
| `""` | 0.5 | PASS | 0 lines |

### 7.6 AlphabeticWordRatioFilter tests (`alphabetic_word_ratio_filter_test.rs`)

Python function: `alphabetic_word_ratio_filter(text, max_ratio=0.2) -> bool`. Words = `text.split()`. Non-alpha word = no character is alphabetic. **Special case: 1 word total or 0 words → FAIL.**

| Input | max_ratio | Expected | Note |
|-------|-----------|----------|------|
| `"This is all alphabetic text"` | 0.2 | PASS | 0/5 = 0.0 |
| `"Some text 123 with 456"` | 0.5 | PASS | 2/5 = 0.4 |
| `"Some text 123 456 789"` | 0.3 | FAIL | 3/5 = 0.6 |
| `"123 456 789"` | 0.5 | FAIL | 3/3 = 1.0 |
| `""` | 0.5 | FAIL | 0 words |

### 7.7 StopWordFilter tests (`stop_word_filter_test.rs`)

Python function: `stop_word_filter(text, count_unique=False, min_stop_word=2) -> bool`. Stop words: `{"the", "be", "to", "of", "and", "that", "have", "with"}`. Case-insensitive. Words via `text.split()`.

| Input | count_unique | min_stop_word | Expected | Note |
|-------|-------------|--------------|----------|------|
| `"This is the document with the important content"` | false | 2 | PASS | "the"×2 + "with"×1 = 3 total |
| `"This is the document with the important content"` | false | 4 | FAIL | only 3 total stop words |
| `"The document and the content that have important information"` | true | 2 | PASS | 4 unique: "the","and","that","have" |
| `"The document with the content"` | true | 3 | FAIL | 2 unique: "the","with" < 3 |
| `"This is THE document with The important content"` | false | 2 | PASS | case-insensitive: "the"×2 + "with"×1 = 3 |
| `""` | false | 2 | FAIL | no stop words |
| `"The document and the content that have important information with details"` | false | 3 | PASS | 5 total: "the"×2,"and","that","have","with" |

### 7.8 MassiveWebRepetitionFilter `_rep_counter_fraction` tests (`massive_repetition_filter_test.rs`)

Python function: `rep_counter_fraction(elements: list[str], ngram_size: int, weighted: bool) -> float`

These test the core algorithm directly — critical for correctness.

| Elements | ngram_size | weighted | Expected | Note |
|----------|-----------|----------|----------|------|
| `[]` | 1 | false | 1.0 | empty input, ngram_size=1 special case |
| `["hello"]` | 1 | false | 0.0 | single element, no repetition |
| `["hello"]` | 2 | false | 0.0 | fewer elements than ngram_size |
| `["a","b","c","d"]` | 1 | false | 0.0 | no repetitions |
| `["a","b","a","c","b","d"]` | 1 | false | 4/6 ≈ 0.667 | "a"×2, "b"×2 = 4 of 6 |
| `["a","a","a","a"]` | 1 | false | 1.0 | all repetitions |
| `["short","looooong","short","medium"]` | 1 | false | 0.5 | "short"×2 = 2 of 4 |
| `["a","b","c","d"]` | 1 | true | 0.0 | no reps, weighted |
| `["aa","bb","aa","cc","bb","dd"]` | 1 | true | 8/12 ≈ 0.667 | "aa"×2(4ch), "bb"×2(4ch), total=12ch |
| `["a","a","a","a"]` | 1 | true | 1.0 | all reps, weighted |
| `["short","looooong","short","medium"]` | 1 | true | 10/24 ≈ 0.417 | "short"×2=10ch, total=5+8+5+6=24 |
| `["a","b","c","d","e"]` | 2 | false | 0.0 | no 2-gram reps |
| `["a","b","c","a","b","d"]` | 2 | false | 4/6 ≈ 0.667 | "a,b"×2, each len 2, total=6 |
| `["a","b","c","d","a","b","c","e"]` | 3 | false | 6/8 = 0.75 | "a,b,c"×2, each len 3, total=8 |
| `["short","a","b","c","short","a","b","d"]` | 4 | false | 0.0 | no 4-gram repeats |
| `["a","a","a","a","a","a","b","c","d","f"]` | 4 | false | 6/10 = 0.6 | overlapping 4-grams of "a" |
| `["a","b","c","d","e","f","g","h"]` | 6 | false | 0.0 | no 6-gram reps |
| `["the","quick","brown","fox","jumps","over","the","lazy","dog", "the","quick","brown","fox","jumps","over","the","lazy","cat"]` | 8 | false | (computed) | 8-gram repeated ×2, compute expected from char lengths |
| `["a","b","c"]` | 3 | false | 0.0 | exactly ngram_size elements, one window |
| `["a","b","c","a","b","c"]` | 3 | false | 1.0 | just over, full repeat |
| `["a"*10000, "b", "a"*10000]` | 1 | true | 20000/20001 | very large strings |

**Realistic text bigram test:**
```python
text = "to be or not to be that is the question to be or not to be I don't know the answer."
elements = text.split()
# ngram_size=2, weighted=false → expected: 16/total_len
# This tests the n≤4 "single most-common repeated n-gram" logic
```

### 7.9 NewlineRemovalModifier tests (`newline_removal_modifier_test.rs`)

Python function: `newline_removal_modifier(text, max_consecutive=2) -> str`

| Input | max_consecutive | Expected |
|-------|----------------|----------|
| `"Hello\n\n\n\nWorld"` | 2 | `"Hello\n\nWorld"` |
| `"Line 1\n\n\n\n\nLine 2\n\n\nLine 3"` | 3 | `"Line 1\n\n\nLine 2\n\n\nLine 3"` |
| `"Hello\n\nWorld\nAgain"` | 2 | `"Hello\n\nWorld\nAgain"` (unchanged) |
| `""` | 2 | `""` |
| `"Line 1\n\n\nLine 2\n\n\n\nLine 3\n\nLine 4"` | 1 | `"Line 1\nLine 2\nLine 3\nLine 4"` |

### 7.10 RatioLineModifier tests (`ratio_line_modifier_test.rs`)

Python function: `ratio_line_modifier(text, upper_bound, check) -> str`. Removes lines where ratio > upper_bound.

**Uppercase mode:**

| Input | upper_bound | Expected lines kept |
|-------|------------|-------------------|
| `"this is a lowercase line\nTHIS IS AN UPPERCASE LINE\nThis Has Some Uppercase Letters\nAnother 50% UPPERCASE line"` | 0.3 | lines 1,3 only |
| `"\n\nThis is a normal line\n\nTHIS IS UPPERCASE\n"` | 0.5 | empty lines pass → `"\n\nThis is a normal line\n\n"` |
| `"HALF uppercase\nAAAaaa\nall lowercase"` | 0.5 | all three |
| `"HALF uppercase\nAAAaaa\nall lowercase"` | 0.0 | `"all lowercase"` only |

**Numeric mode:**

| Input | upper_bound | Expected lines kept |
|-------|------------|-------------------|
| `"This is a text without numbers\nThis has 1 number\nThis has 12345 numbers\nPhone: 555-123-4567"` | 0.2 | lines 1,2 only |

### 7.11 RegexLineModifier tests (`regex_line_modifier_test.rs`)

Python function: `regex_line_modifier(text, regex=DEFAULT_SOCIAL_REGEX) -> str | None`. Returns None if all lines removed.

Default regex matches social media counters (case-insensitive).

| Input | Expected |
|-------|----------|
| `"This is a normal line\n10K likes\nAnother normal line\n5.2M views\nFinal normal line"` | `"This is a normal line\nAnother normal line\nFinal normal line"` |
| `"This is a normal line\nAnother normal line\nFinal normal line"` | unchanged |
| `"10K likes\n5.2M views\n3B followers"` | None |
| `"This is a normal line\n10k Likes\nAnother normal line\n5.2m Views"` | `"This is a normal line\nAnother normal line"` |
| `"1k likes\n1.5K likes\n10,000 followers\n(10M views)\n 5B downloads "` | None |

### 7.12 LineLenModifier tests (`line_len_modifier_test.rs`)

Python function: `line_len_modifier(text, lower_bound=0) -> str | None`. Removes lines with <lower_bound words (using `unicode_words()` equivalent). Returns None if no lines survive.

| Input | lower_bound | Expected |
|-------|------------|----------|
| `""` | 1 | None |
| `"hello\nhi there"` | 3 | None |
| `"hello world\nrust is awesome"` | 2 | unchanged |
| `"hello world\nrust is awesome\nshort line"` | 3 | `"rust is awesome"` |
| `"hello world\nsingle"` | 2 | `"hello world"` |
| `"こんにちは 世界 rust\nHello world"` | 3 | `"こんにちは 世界 rust"` |
| `"hello\n\nempty line"` | 0 | unchanged |

### 7.13 SubstringLineModifier tests (`substring_line_modifier_test.rs`)

Python function: `substring_line_modifier(text, banlist, max_len=MAX, remove_substring_only=True, location="any") -> str`

| Input | banlist | remove_sub_only | location | max_len | Expected |
|-------|---------|----------------|----------|---------|----------|
| `"This is a bad line.\nThis is a good line.\nThis is a badger line."` | `"bad"` | true | any | MAX | `"This is a  line.\nThis is a good line.\nThis is a ger line."` |
| (same) | `"bad"` | false | any | MAX | `"This is a good line."` |
| `"This is bad line.\nShort line.\nThis is a good long line."` | `"bad"` | false | any | 4 | `"Short line.\nThis is a good long line."` |
| `"bad start of line.\nMiddle bad word.\nbadger beginning.\nNormal line."` | `"bad"` | true | prefix | MAX | `" start of line.\nMiddle bad word.\nger beginning.\nNormal line."` |
| `"End of line bad\nMiddle bad word.\nEnding in bad\nNormal line."` | `"bad"` | true | suffix | MAX | `"End of line \nMiddle bad word.\nEnding in \nNormal line."` |
| `"bad\nThis is good.\nbad bad"` | `"bad"` | true | any | MAX | `"This is good."` |
| `"This is very bad\nThis is bad only.\nThis is very very bad indeed."` | `"very bad"` | true | any | MAX | `"This is \nThis is bad only.\nThis is very  indeed."` |
| `"This contains Bad.\nThis contains bad."` | `"Bad"` | true | any | MAX | `"This contains .\nThis contains bad."` (case-sensitive) |

### 7.14 WordRemovalRatioFilter tests (`word_removal_ratio_filter_test.rs`)

Python function: `word_removal_ratio_filter(text, original_word_count, upper_bound=1.0) -> bool`. PASS if removal ratio ≤ upper_bound.

| text (current) | original_word_count | upper_bound | Expected | Note |
|----------------|-------------------|------------|----------|------|
| 8-word text | 10 | 0.3 | PASS | 20% removed |
| 7-word text | 10 | 0.3 | PASS | 30% = bound |
| 6-word text | 10 | 0.3 | FAIL | 40% > 30% |
| 10-word text | 10 | 0.3 | PASS | 0% removed |

### 7.15 Cross-validation with Rust binary (optional)

For maximum confidence, the test suite can optionally run the actual `datamap-rs` binary on the same inputs and compare outputs. This requires cloning and building datamap-rs.

```python
# conftest.py
import subprocess, json, pytest, os

DATAMAP_BINARY = os.environ.get("DATAMAP_RS_BINARY")

@pytest.fixture
def datamap_rs():
    """Skip tests if datamap-rs binary not available."""
    if not DATAMAP_BINARY or not os.path.isfile(DATAMAP_BINARY):
        pytest.skip("datamap-rs binary not available (set DATAMAP_RS_BINARY)")
    return DATAMAP_BINARY
```

This is nice-to-have for CI but not required for the initial implementation. The ported test vectors above provide sufficient coverage for a faithful Python reimplementation.
